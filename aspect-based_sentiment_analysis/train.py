# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_helper import MultiClsDataSet
from sklearn.metrics import accuracy_score
from data_sentiment import Sentiment
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
import pandas as pd

train_path = "./data/train.json"
dev_path = "./data/dev.json"
test_path = "./data/test.json"
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
class_num = len(label2idx)  # 11个类别
label_num = 3  # 0-正向 1-中性 2-负向
att_size = 256 #Bert输出后，初始化维度，供attention使用
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 64
max_len = 128
hidden_size = 1024
epochs = 10

train_dataset = MultiClsDataSet(train_path, max_len=max_len, label2idx_path=label2idx_path)
dev_dataset = MultiClsDataSet(dev_path, max_len=max_len, label2idx_path=label2idx_path)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)


def get_acc_score(y_true_tensor, y_pred_tensor):
    # y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_pred_tensor = y_pred_tensor.argmax(dim=2).cpu().detach().numpy()
    y_pred_tensor = y_pred_tensor.flatten()
    y_true_tensor = y_true_tensor.cpu().numpy()
    y_true_tensor = y_true_tensor.flatten()
    return accuracy_score(y_true_tensor, y_pred_tensor)


def train():
    sentiment_output = Sentiment.encoder()[1].to(device)

    model = BertMultiLabelCls(hidden_size=hidden_size, label_num=label_num, class_num=class_num, att_size=att_size, sentiment_output=sentiment_output)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    dev_best_acc = 0.

    for epoch in range(1, epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            labels = batch[-1]
            logits = model(*batch[:3])
            loss = 0
            for x in range(class_num):
                loss = loss + criterion(logits[:, x], labels[:, x])
            loss.backward()
            optimizer.step()

            # if i % 10 == 0:
            acc_score = get_acc_score(labels, logits)
            print("Train epoch:{} step:{}  acc: {} loss:{} ".format(epoch, i, acc_score, loss.item()))

        # 验证集合
        dev_loss, dev_acc = dev(model, dev_dataloader, criterion)
        print("Dev epoch:{} acc:{} loss:{}".format(epoch, dev_acc, dev_loss))
        if dev_acc > dev_best_acc:
            dev_best_acc = dev_acc
            torch.save(model.state_dict(), save_model_path)

    # 测试
    test(save_model_path, test_path)



def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = 0
            for x in range(class_num):
                loss = loss + criterion(logits[:, x], labels[:, x])
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return np.mean(all_loss), acc_score


def test(model_path, test_data_path):
    test_dataset = MultiClsDataSet(test_data_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    sentiment_output = Sentiment.encoder()[1].to(device)
    model = BertMultiLabelCls(hidden_size=hidden_size, label_num=label_num, class_num=class_num, att_size=att_size, sentiment_output=sentiment_output)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            true_labels.append(labels)
            pred_labels.append(logits)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)

    acc_score = get_acc_score(true_labels, pred_labels)
    # 计算AUC和F1分数
    auc_score = get_auc_score(true_labels, pred_labels)
    f1_score = get_f1_score(true_labels, pred_labels)

    print("Acc Score:", acc_score)
    print("AUC Score:", auc_score)
    print("F1 Score:", f1_score)
    return acc_score,auc_score,f1_score




def get_auc_score(y_true, y_pred_log):
    # 将对数概率转换回概率
    y_pred = torch.exp(y_pred_log).numpy()

    n_samples = y_true.shape[0]
    auc_scores = []

    for index in range(n_samples):
        y_true_sample = y_true[index]
        y_pred_sample = y_pred[index]

        # 对每个分类的标签进行二值化
        y_true_bin = label_binarize(y_true_sample, classes=[0, 1, 2])

        # 计算每个类的AUC并取平均
        sample_auc_scores = []
        for i in range(y_true_bin.shape[1]):  # 遍历每个标签
            try:
                auc_score = roc_auc_score(y_true_bin[:, i], y_pred_sample[:, i])
                sample_auc_scores.append(auc_score)
            except ValueError:
                continue  # 忽略只有一类的标签

        if sample_auc_scores:
            auc_scores.append(np.mean(sample_auc_scores))

    return np.mean(auc_scores) if auc_scores else 0


def get_f1_score(y_true, y_pred_log):
    # 取argmax获取预测类别
    y_pred = y_pred_log.argmax(dim=2).numpy()

    n_samples = y_true.shape[0]
    f1_scores = []

    for index in range(n_samples):
        y_true_sample = y_true[index]
        y_pred_sample = y_pred[index]

        # 计算F1分数
        f1_scores.append(f1_score(y_true_sample, y_pred_sample, average='macro'))

    return np.mean(f1_scores)


if __name__ == '__main__':
    test(save_model_path, test_path)
