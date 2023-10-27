import pandas
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from data_sentiment import Sentiment
from data_csv_dataset import MultiClsCSVDataSet
import glob

train_path = "./data/train.json"
dev_path = "./data/dev.json"
test_path = "./data/test.json"
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
class_num = len(label2idx)  # 11个类别
label_num = 3  # 0-正向 1-中性 2-负向
att_size = 256  # Bert输出后，初始化维度，供attention使用
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-5
batch_size = 64
max_len = 128
hidden_size = 1024


def predict_csv(model_path, data_path):
    test_dataset = MultiClsCSVDataSet(data_path, max_len=max_len, label2idx_path=label2idx_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    sentiment_output = Sentiment.encoder()[1].to(device)
    model = BertMultiLabelCls(hidden_size=hidden_size, label_num=label_num, class_num=class_num, att_size=att_size,
                              sentiment_output=sentiment_output)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    pred_labels = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_mask, token_type_ids, labels = [d.to(device) for d in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = logits.argmax(dim=2).cpu().detach().numpy()
            pred_labels.extend(logits)

    return pred_labels


def read_csv(data_path):
    file_names = glob.glob(data_path + '/*.csv')
    df = pandas.DataFrame()
    total = []
    label2value = [1, 0, -1]  # 将 0 , 1 , 2 分别转换成1 , 0 , -1
    # 打印文件名称
    for file_name in file_names:
        product_name = file_name.split("_")[0]
        print(product_name)
        sale_count = int(file_name.split("_")[1])
        print(sale_count)
        pred_labels = predict_csv(save_model_path, file_name)
        # pred_labels = [[0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2,2,2]]

        # 将 0 , 1 , 2 分别转换成 1 , 0 , -1 ; 0-正向 1-中性 2-负向
        for j in range(pred_labels.__len__()):
            pred_labels[j] = [label2value[i] for i in pred_labels[j]]

        tmp = pandas.DataFrame(pred_labels)
        tmp.to_excel('./data/tmp/{}.xlsx'.format(product_name.split('/')[3]), header=False)
        line = []
        line.append(product_name)
        for i in range(tmp.shape[1]):
            score = tmp.iloc[:, i].sum() * 1.0 / tmp.iloc[:, i].abs().sum()
            score = F.sigmoid(torch.tensor(score)).numpy().item()
            line.append(score)
        line.append(sale_count)

        total.append(line)

    df = pandas.DataFrame(total)

    df = df.fillna(0.5)  # 默认值为0，取sigmoid后为0.5

    df.to_excel('./data/score.xlsx', header=False)


def statistics_tmp(data_path):
    file_names = glob.glob(data_path + '/*.xlsx')
    df = pandas.DataFrame()
    for file_name in file_names:
        df1 = pandas.read_excel(file_name, header=None, index_col=0)
        df = pandas.concat([df, df1], axis=0)
    df_stat = pandas.DataFrame([[0]*11]*1)
    for x in range(df.shape[1]):
        df_stat.iloc[0, x] = df.iloc[:, x][df.iloc[:, x] != 0].count()
    df_stat.to_excel('./data/stat.xlsx')

    print()

def statistics_csv(data_path):
    file_names = glob.glob(data_path + '/*.csv')
    df = pandas.DataFrame()
    for file_name in file_names:
        df1 = pandas.read_csv(file_name)
        df = pandas.concat([df, df1], axis=0)
    print()

if __name__ == '__main__':
    # read_csv(data_path="./data/csv")
    statistics_csv(data_path="./data/csv")
