import json
from torch.utils.data import Dataset, DataLoader
import jieba
import fasttext.util
import torch


# 加载预训练的fastText模型
model = fasttext.load_model('../../model/fasttext/cc.zh.300.bin')
fasttext.util.reduce_model(model, 300)

def text_to_embedding(text):
    """
    将文本转换为128维的嵌入向量。

    参数:
    - text: 要转换的文本，字符串类型。
    - model: 已加载的fastText模型。

    返回:
    - embedding: 文本的100维嵌入向量，numpy数组。
    """
    # 使用jieba进行分词
    words = list(jieba.cut(text))

    vectors_zero = [torch.rand(1, model.get_dimension()) for word in words if word.strip()]  # 假设每个词向量维度为10

    # 计算分词后的所有词向量的平均值
    vectors = [torch.tensor(model.get_word_vector(word)).unsqueeze(0) for word in words if word.strip()]
    if vectors:
        embedding = vectors
    else:
        embedding = vectors_zero # 如果文本中没有有效词，则返回128维零向量

    embedding = torch.cat(embedding, dim=0)
    return embedding
# 定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.label2idx = self.load_json("../data/label2idx.json")
        self.class_num = len(self.label2idx)
        self.texts,self.labels = self.encoder(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        if self.transform:
            text = self.transform(text)
        return text, label

    def load_json(self,data_path):
        with open(data_path, encoding="utf-8") as f:
            return json.loads(f.read())
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
        return texts, labels