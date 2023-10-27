# -*- coding: utf-8 -*-

import torch
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import BertTokenizer
from data_sentiment import Sentiment

hidden_size = 1024
class_num = 3
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
idx2label = {idx: label for label, idx in label2idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
label_num = 3  # 0-正向 1-中性 2-负向
att_size = 256 #Bert输出后，初始化维度，供attention使用
MODEL_PATH = r"../model/chinese_roberta_wwm_large_ext_pytorch"
# 导入模型
tokenizer = BertTokenizer.from_pretrained(
    r"../model/chinese_roberta_wwm_large_ext_pytorch/vocab.txt")


max_len = 128

sentiment_output = Sentiment.encoder()[1].to(device)
model = BertMultiLabelCls(hidden_size=hidden_size, label_num=label_num, class_num=class_num, att_size=att_size,
                          sentiment_output=sentiment_output)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
model.eval()


def predict(texts):
    outputs = tokenizer(texts, return_tensors="pt", max_length=max_len,
                        padding=True, truncation=True)
    logits = model(outputs["input_ids"].to(device),
                   outputs["attention_mask"].to(device),
                   outputs["token_type_ids"].to(device))
    logits = logits.argmax(dim=2).cpu().tolist()
    # print(logits)
    result = []
    i = 0
    for sample in logits:
        print(texts[i])
        pred_label = []
        for idx, logit in enumerate(sample):
            if logit != 1:
                if logit == 0:
                    print(idx2label[idx]  + ":positive")
                elif logit == 2:
                    print(idx2label[idx]  + ":negative")
        i = i + 1
    return result


if __name__ == '__main__':
    texts = ["有点儿贵，不太新鲜，下次不会在买了，快递很慢", "不错正是想要买的，配料表很干净适合减脂期，果仁品质很高吃起来味道很好，最重要的是原味对减脂的人群太友好了，每天早晨作为优质脂肪来补充，等吃完了还会来回购，快递很慢，猫超服务态度太差"]
    predict(texts)


