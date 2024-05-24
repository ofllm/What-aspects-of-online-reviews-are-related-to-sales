import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset, text_to_embedding
from rnnmodel import RNNModel
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

bidirectional = True
model_name = "lstm"

epochs = 50
label_num = 3
class_num = 11
dropout = 0.5
lr = 1e-5
input_size = 300
hidden_size = 256
train_path = "../data/train.json"
dev_path = "../data/dev.json"
test_path = "../data/test.json"
label2idx_path = "../data/label2idx.json"
if bidirectional:
    save_model_path = "multi_label_cls_" + model_name + "_bidirectional.pth"
else:
    save_model_path = "multi_label_cls_" + model_name + ".pth"
batch_size = 128
num_layers = 10


# 数据加载器的collate_fn
def collate_fn(batch):
    batch_x = [x for x, _ in batch]
    batch_y = [y for _, y in batch]
    lengths = [len(x) for x in batch_x]
    batch_x_padded = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)
    return batch_x_padded, torch.tensor(batch_y), torch.tensor(lengths)

# 创建模型实例
model = RNNModel(input_size, hidden_size, num_layers, label_num, dropout, bidirectional, model_name)

# 判断是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 创建Dataloader
train_dataset = CustomDataset(train_path,transform=text_to_embedding)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

dev_dataset = CustomDataset(dev_path,transform=text_to_embedding)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

test_dataset = CustomDataset(test_path,transform=text_to_embedding)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def get_acc_score(y_true_tensor, y_pred_tensor):
    # y_pred_tensor = (y_pred_tensor.cpu() > 0.5).int().numpy()
    y_pred_tensor = y_pred_tensor.argmax(dim=2).cpu().detach().numpy()
    y_pred_tensor = y_pred_tensor.flatten()
    y_true_tensor = y_true_tensor.cpu().numpy()
    y_true_tensor = y_true_tensor.flatten()
    return accuracy_score(y_true_tensor, y_pred_tensor)



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

# 定义训练函数
def train(model, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    best_acc = 0.0

    for epoch in range(epochs):  # 假设训练100个epoch
        model.train()
        current = 0
        for inputs, labels, lengths in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs,lengths)
            loss = 0
            for x in range(class_num):
                loss = loss + criterion(outputs[:, x], labels[:, x])
            loss.backward()
            optimizer.step()
            acc_score = get_acc_score(labels, outputs)
            print("Train epoch:{} step:{} acc: {} loss:{} ".format(epoch, current, acc_score, loss.item()))
            current = current + 1
        # 在每个epoch结束后进行验证
        # 验证集合
        dev_loss, dev_acc = dev(model, dev_dataloader, criterion)
        print("Dev epoch:{} acc:{} loss:{}".format(epoch, dev_acc, dev_loss))
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), save_model_path)

def dev(model, dataloader, criterion):
    all_loss = []
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs,lengths)
            loss = 0
            for x in range(class_num):
                loss = loss + criterion(outputs[:, x], labels[:, x])
            all_loss.append(loss.item())
            true_labels.append(labels)
            pred_labels.append(outputs)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    return np.mean(all_loss), acc_score


def test(test_loader):
    model_predict = RNNModel(input_size, hidden_size, num_layers, label_num, dropout, bidirectional, model_name)
    # 加载保存的状态字典
    state_dict = torch.load(save_model_path)
    # 将状态字典应用到模型实例
    model_predict.load_state_dict(state_dict)
    model_predict.to(device)
    model_predict.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels, lengths in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_predict(inputs,lengths)
            true_labels.append(labels)
            pred_labels.append(outputs)
    true_labels = torch.cat(true_labels, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    acc_score = get_acc_score(true_labels, pred_labels)
    print("Test acc: {}".format(acc_score))
    return

# 训练模型
if __name__ == '__main__':
    print(model_name + "模型训练开始... bidirectional:" + bidirectional.__str__())
    train(model, train_loader)
    test(test_dataloader)
    print(model_name + "模型训练结束...")
