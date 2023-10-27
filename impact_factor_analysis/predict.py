import copy
import os
import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import DataLoader

from net import Net01
import dataset
import pandas as pd

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络
    net = Net01().to(device)  # 全连接网络
    # 优化器
    opt = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    # 损失函数
    mseLoss = nn.MSELoss(reduction="mean")
    # 加载预训练权重
    netFilename = r"./model/net_sales.pth"
    net.load_state_dict(torch.load(netFilename))

    # 训练集 和 测试集
    predict_datasets = dataset.SalesDataset(datafile="./data/predict.xls", isTrain=False ,isPredict=True)
    predict_dataloader = DataLoader(predict_datasets, batch_size=200, shuffle=False)

    net.eval()
    for i, (data, target) in enumerate(predict_dataloader):
        df = pd.DataFrame([[0.0]*11]*2,columns=["品质", "口味", "新鲜", "份量", "大小", "包装", "价格", "客服", "快递", "回购", "推荐"])
        for j in range(data.shape[1]):
            miv = [[0.95, 1.05],[0.9, 1.1]]
            for x in range(2):
                data1 = copy.deepcopy(data)
                data1[:, j] = data1[:, j] * miv[x][0]
                data1, _ = data1.to(device), target.to(device)
                pred1 = net(data1)

                data2 = copy.deepcopy(data)
                data2[:, j] = data2[:, j] * miv[x][1]
                data2, _ = data2.to(device), target.to(device)
                pred2 = net(data2)
                pred =(pred2-pred1).abs().mean()

                df.iloc[x, j] = pred.detach().cpu().numpy()

        df.to_excel('./data/out.xlsx')



