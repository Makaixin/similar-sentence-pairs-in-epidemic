from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader

from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score


parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--seed", type=int, default=465, help="seed")
parser.add_argument("--cnt", type=int, default=0)
parser.add_argument("--cut", type=int, default=-1)

arg = parser.parse_args()

device = torch.device('cuda')

train = pd.read_csv('../data/Dataset/train.csv')

dev = pd.read_csv('../data/Dataset/dev.csv')

train_category = []
train_query1 = []
train_query2 = []
train_label = []

for i in range(len(train)):
    if train['label'][i] != train['label'][i]:
        continue
    train_category.append(train['category'][i])
    train_query1.append(train['query1'][i])
    train_query2.append(train['query2'][i])
    train_label.append(train['label'][i])

for i in range(len(dev)):
    if dev['label'][i] != dev['label'][i]:
        continue
    train_category.append(dev['category'][i])
    train_query1.append(dev['query1'][i])
    train_query2.append(dev['query2'][i])
    train_label.append(dev['label'][i])

train_category = np.array(train_category)
train_query1 = np.array(train_query1)
train_query2 = np.array(train_query2)
train_label = np.array(train_label).astype(int)

model_path = '../user_data/pre-trained/chinese_roberta_wwm_large_ext_pytorch/'

bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)


class BertForClass(nn.Module):
    def __init__(self, n_classes=2):
        super(BertForClass, self).__init__()
        self.model_name = 'BertForClass'
        self.bert_model = BertModel.from_pretrained(model_path, config=bert_config)
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)

    def forward(self, input_ids, input_masks, segment_ids):
        sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        logit = self.classifier(concat_out)

        return logit


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class data_generator:
    def __init__(self, data, batch_size=16, max_length=64, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        c, q1, q2, y = self.data
        idxs = list(range(len(self.data[0])))
        if self.shuffle:
            np.random.shuffle(idxs)
        input_ids, input_masks, segment_ids, labels = [], [], [], []

        for index, i in enumerate(idxs):

            text = q1[i]
            text_pair = q2[i]

            input_id = tokenizer.encode(text, text_pair, max_length=self.max_length)
            input_mask = [1] * len(input_id)
            if len(text) + 2 <= len(input_id):
                segment_id = [0] * (len(text) + 2) + [1] * (len(input_id) - 2 - len(text))
            else:
                segment_id = [0] * len(input_id)
            padding_length = self.max_length - len(input_id)
            input_id += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_id += ([0] * padding_length)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(y[i])
            if len(input_ids) == self.batch_size or i == idxs[-1]:
                yield input_ids, input_masks, segment_ids, labels
                input_ids, input_masks, segment_ids, labels = [], [], [], []


skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=arg.seed)

epoch = 10

for fold, (train_index, valid_index) in enumerate(skf.split(train_category, train_label)):
    if fold <= arg.cut:
        continue
    print('\n\n------------fold:{}------------\n'.format(fold))
    c = train_category[train_index]
    q1 = train_query1[train_index]
    q2 = train_query2[train_index]
    y = train_label[train_index]

    val_c = train_category[valid_index]
    val_q1 = train_query1[valid_index]
    val_q2 = train_query2[valid_index]
    val_y = train_label[valid_index]

    train_D = data_generator([c, q1, q2, y], batch_size=1, shuffle=True)
    val_D = data_generator([val_c, val_q1, val_q2, val_y], batch_size=1)

    model = BertForClass().to(device)
    pgd = PGD(model)
    K = 3
    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(params=model.parameters(), lr=1e-5)

    best_acc = 0
    PATH = '../user_data/model_data/bert_{}.pth'.format(fold+arg.cnt)
    for e in range(epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D)

        for input_ids, input_masks, segment_ids, labels in tq:
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            y_pred = model(input_ids, input_masks, segment_ids)

            loss = loss_fn(y_pred, label_t)
            loss.backward()
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                y_pred = model(input_ids, input_masks, segment_ids)

                loss_adv = loss_fn(y_pred, label_t)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            # 梯度下降，更新参数
            optimizer.step()
            model.zero_grad()

            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            acc += sum(y_max == labels)
            loss_num += loss.item()
            train_len += len(labels)
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)


        model.eval()
        y_p = []
        for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            y_pred = model(input_ids, input_masks, segment_ids)

            y_max = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            y_p.append(y_max[0])

        acc = 0
        for i in range(len(y_p)):
            if val_y[i] == y_p[i]:
                acc += 1
        acc = acc / len(y_p)
        print("best_acc:{}  acc:{}\n".format(best_acc, acc))
        if acc >= best_acc:
            best_acc = acc
            torch.save(model, PATH)

    optimizer.zero_grad()
