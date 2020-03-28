from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
import datetime

device = torch.device('cuda')
test = pd.read_csv('../data/Dataset/test.csv')

model_path = '../user_data/pre-trained/chinese_roberta_wwm_large_ext_pytorch/'

test_category = test['category'].values
test_query1 = test['query1'].values
test_query2 = test['query2'].values

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


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


oof_test = np.zeros((len(test), 2), dtype=np.float32)
vote = np.zeros((len(test), 2), dtype=np.float32)

test_c = test_category[range(len(test_category))]
test_q1 = test_query1[range(len(test_category))]
test_q2 = test_query2[range(len(test_category))]
test_i = test_category[range(len(test_category))]

test_D = data_generator([test_c, test_q1, test_q2, test_i], batch_size=1)

f = 11

for fold in range(f):

    PATH = '../user_data/model_data/bert_{}.pth'.format(fold)

    model = torch.load(PATH).to(device)
    model.eval()
    y_p = []
    for input_ids, input_masks, segment_ids, labels in tqdm(test_D):
        input_ids = torch.tensor(input_ids).to(device)
        input_masks = torch.tensor(input_masks).to(device)
        segment_ids = torch.tensor(segment_ids).to(device)

        y_pred = model(input_ids, input_masks, segment_ids)

        y_p += y_pred.detach().to("cpu").tolist()
    y_p = np.array(y_p)
    y_p = softmax(y_p)
    oof_test += y_p
    for i in range(len(y_p)):
        vote[i][np.argmax(y_p[i])] += 1
    # np.savetxt('./bert_{}.txt'.format(fold), y_p)

for i in range(len(test)):
    vote[i][np.argmax(oof_test[i])] += 3.5
# np.savetxt('./v.txt', vote)
test['label'] = np.argmax(vote, axis=1)
test[['id', 'label']].to_csv('../prediction_result/result'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv', index=False)
