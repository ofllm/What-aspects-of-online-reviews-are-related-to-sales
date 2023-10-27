# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from layer import ScaledDotProductAttention


class BertMultiLabelCls(nn.Module):
    def __init__(self, hidden_size, label_num, class_num, att_size, sentiment_output, dropout=0.1):
        super(BertMultiLabelCls, self).__init__()
        self.sentiment_output = sentiment_output
        self.init = nn.Linear(hidden_size, hidden_size)


        self.fc0 = nn.Linear(hidden_size, label_num)
        self.fc1 = nn.Linear(hidden_size, label_num)
        self.fc2 = nn.Linear(hidden_size, label_num)
        self.fc3 = nn.Linear(hidden_size, label_num)
        self.fc4 = nn.Linear(hidden_size, label_num)
        self.fc5 = nn.Linear(hidden_size, label_num)
        self.fc6 = nn.Linear(hidden_size, label_num)
        self.fc7 = nn.Linear(hidden_size, label_num)
        self.fc8 = nn.Linear(hidden_size, label_num)
        self.fc9 = nn.Linear(hidden_size, label_num)
        self.fc10 = nn.Linear(hidden_size, label_num)

        self.drop = nn.Dropout(dropout)

        MODEL_PATH = r"../model/chinese_roberta_wwm_large_ext_pytorch"
        # 导入配置文件
        model_config = transformers.BertConfig.from_pretrained(MODEL_PATH)
        # 修改配置
        model_config.output_hidden_states = True
        model_config.output_attentions = True
        # 通过配置和路径导入模型
        self.bert = transformers.BertModel.from_pretrained(MODEL_PATH, config=model_config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls = self.drop(outputs[1])

        cls = self.init(cls)

        att = ScaledDotProductAttention(5)
        cls = cls.reshape(cls.size()[0], 32, 32)

        q = torch.stack([self.sentiment_output[0] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls0, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[1] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls1, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[2] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls2, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[3] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls3, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[4] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls4, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[5] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls5, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[6] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls6, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[7] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls7, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[8] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls8, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[9] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls9, _ = att(q, cls, cls)

        q = torch.stack([self.sentiment_output[10] for i in range(cls.size()[0])], dim=0)
        q = q.reshape(q.size()[0], 32, 32)
        cls10, _ = att(q, cls, cls)

        cls0 = cls.reshape(cls0.size()[0], 1024)
        cls1 = cls.reshape(cls1.size()[0], 1024)
        cls2 = cls.reshape(cls2.size()[0], 1024)
        cls3 = cls.reshape(cls3.size()[0], 1024)
        cls4 = cls.reshape(cls4.size()[0], 1024)
        cls5 = cls.reshape(cls5.size()[0], 1024)
        cls6 = cls.reshape(cls6.size()[0], 1024)
        cls7 = cls.reshape(cls7.size()[0], 1024)
        cls8 = cls.reshape(cls8.size()[0], 1024)
        cls9 = cls.reshape(cls9.size()[0], 1024)
        cls10 = cls.reshape(cls10.size()[0], 1024)

        out0 = F.log_softmax(self.fc0(cls0), dim=1)
        out1 = F.log_softmax(self.fc1(cls1), dim=1)
        out2 = F.log_softmax(self.fc2(cls2), dim=1)
        out3 = F.log_softmax(self.fc3(cls3), dim=1)
        out4 = F.log_softmax(self.fc4(cls4), dim=1)
        out5 = F.log_softmax(self.fc5(cls5), dim=1)
        out6 = F.log_softmax(self.fc6(cls6), dim=1)
        out7 = F.log_softmax(self.fc7(cls7), dim=1)
        out8 = F.log_softmax(self.fc8(cls8), dim=1)
        out9 = F.log_softmax(self.fc9(cls9), dim=1)
        out10 = F.log_softmax(self.fc10(cls10), dim=1)

        out = torch.stack((out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10), dim=1)

        return out
