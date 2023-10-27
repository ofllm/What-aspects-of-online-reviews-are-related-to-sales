import os
import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import DataLoader

from net import Net01
import dataset
import val

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
    # net.load_state_dict(torch.load(netFilename)) if os.path.exists(netFilename) else ...

    # 训练集 和 测试集
    train_datasets = dataset.SalesDataset(datafile="./data/impact_factor_analysis.xls", isTrain=True)
    test_datasets = dataset.SalesDataset(datafile="./data/impact_factor_analysis.xls", isTrain=False)

    train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False)

    test_best_mse = 100000.
    # 训练
    for epoch in range(100000):
        net.train()  # 开启训练
        train_losses = []
        for i, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            pred = net(data)
            loss = mseLoss(pred, target)

            # 梯度清空 反向传播 更新梯度
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss)

        train_losses = torch.tensor(train_losses)
        train_mean_loss = torch.mean(train_losses)

        if epoch % 1000 == 0:
            print("epoch:{},train loss = ".format(epoch), train_mean_loss.detach().cpu().item())
            net.eval()
            test_losses = []
            mse_s, r2_s, evs_s = [], [], []
            for i, (data, target) in enumerate(test_dataloader):
                data, target = data.to(device), target.to(device)
                pred = net(data)
                loss = mseLoss(pred, target)

                target, pred = target.detach().cpu(), pred.detach().cpu()
                mse, r2, evs = val.reg_calculate(target, pred)
                mse_s.append(mse)
                r2_s.append(r2)
                evs_s.append(evs)

                test_losses.append(loss)

            test_losses = torch.tensor(test_losses)
            test_mean_loss = torch.mean(test_losses)
            print("-----test loss= ", test_mean_loss.detach().cpu().item())

            mse_s, r2_s, evs_s = torch.tensor(mse_s), torch.tensor(r2_s), torch.tensor(evs_s)
            mse_mean, r2_mean, evs_mean = torch.mean(mse_s), torch.mean(r2_s), torch.mean(evs_s)
            mse_mean, r2_mean, evs_mean = mse_mean.detach().cpu().item(), r2_mean.detach().cpu().item(), evs_mean.detach().cpu().item()

            print("-----mse:{0}-----r2:{1}-----evs:{2}".format(mse_mean, r2_mean, evs_mean))

            if mse_mean < test_best_mse:
                test_best_mse = mse_mean
                torch.save(net.state_dict(), "./model/net_sales.pth")
