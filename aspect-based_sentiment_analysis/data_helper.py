# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
import transformers
from data_preprocess import load_json


class MultiClsDataSet(Dataset):
    def __init__(self, data_path, max_len=128, label2idx_path="./data/label2idx.json"):
        self.label2idx = load_json(label2idx_path)
        self.class_num = len(self.label2idx)

        MODEL_PATH = r"../model/chinese_roberta_wwm_large_ext_pytorch"
        # 导入模型

        self.tokenizer =  transformers.BertTokenizer.from_pretrained(r"../model/chinese_roberta_wwm_large_ext_pytorch/vocab.txt")
        self.max_len = max_len
        self.input_ids, self.token_type_ids, self.attention_mask, self.labels = self.encoder(data_path)

    def encoder(self, data_path):
        texts = []
        labels = []
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        for key in data.keys():
            texts.append(data[key]["text"])
            tmp_label = [1] * self.class_num
            for label in data[key]["label"].keys():
                tmp_label[self.label2idx[label]] = data[key]["label"][label]
            labels.append(tmp_label)

        tokenizers = self.tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt",
                                    is_split_into_words=False)
        input_ids = tokenizers["input_ids"]
        token_type_ids = tokenizers["token_type_ids"]
        attention_mask = tokenizers["attention_mask"]

        return input_ids, token_type_ids, attention_mask, \
               torch.tensor(labels, dtype=torch.long)

    def sentiment_encoder(self):
        sentences = ["品质", "口味", "新鲜", "份量", "大小", "包装", "价格", "客服", "快递", "回购", "推荐"]
        tokenizers = self.tokenizer(sentences, padding=True, return_tensors='pt')
        input_ids = tokenizers["input_ids"]
        token_type_ids = tokenizers["token_type_ids"]
        attention_mask = tokenizers["attention_mask"]

        return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.input_ids[item],  self.attention_mask[item], \
               self.token_type_ids[item], self.labels[item]


if __name__ == '__main__':
    dataset = MultiClsDataSet(data_path="./data/train.json")
    print(dataset.input_ids)
    print(dataset.token_type_ids)
    print(dataset.attention_mask)
    print(dataset.labels)