import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_model
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import jieba
import json
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

model_path = '../../model/fasttext/cc.zh.300.bin'
def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, path='./word2vec.model'):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.save(path)

def load_word2vec(path='./word2vec.model'):
    return load_facebook_model(model_path)


def calculate_accuracy(y_true, y_pred):
    accuracies = []
    for i in range(y_true.shape[1]):  # 遍历每个输出
        accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    accuracy = np.mean(accuracies)
    return accuracy

def calculate_auc(y_true, y_pred_log):
    # 初始化一个列表来存储每个类别的 AUC
    overall_auc_scores = []

    # 遍历每个类别
    for i in range(y_true.shape[1]):
        # 获取当前类别的真实标签
        true_labels = y_true[:, i]

        # 获取当前类别的预测概率矩阵
        pred_probs = y_pred_log[i]

        # 初始化一个列表来存储当前类别的每个标签的 AUC
        class_auc_scores = []

        # 遍历每个标签
        for label_index in range(3):
            # 获取当前标签的预测概率
            label_probs = pred_probs[:, label_index]

            # 计算当前标签的 AUC
            # 重新生成二进制标签，因为每次关注的标签都不同
            binary_labels = (true_labels == label_index).astype(int)
            if len(np.unique(binary_labels)) > 1:  # 确保至少有两个类别存在
                auc = roc_auc_score(binary_labels, label_probs)
                class_auc_scores.append(auc)

        # 计算当前类别的所有标签的 AUC 的平均值
        if class_auc_scores:
            avg_auc = np.mean(class_auc_scores)
            overall_auc_scores.append(avg_auc)

    # 计算所有类别的 AUC 的平均值
    if overall_auc_scores:
        final_avg_auc = np.mean(overall_auc_scores)
        return final_avg_auc
    else:
        print("No valid AUC scores could be computed.")

def calculate_f1(y_true, y_pred):
    n_samples = y_true.shape[0]
    f1_scores = []

    for index in range(n_samples):
        y_true_sample = y_true[index]
        y_pred_sample = y_pred[index]

        # 计算F1分数
        f1_scores.append(f1_score(y_true_sample, y_pred_sample, average='macro'))

    return np.mean(f1_scores)


def load_data(path='data.json'):
    with open(path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data).T
    df = df.reset_index(drop=True)
    df['text'] = df['text'].apply(jieba.lcut)
    labels = pd.DataFrame(df['label'].to_list())
    df = pd.concat([df['text'], labels], axis=1)
    return df

def to_vector(words, word_vectors, vector_size):
    vector = np.zeros(vector_size)
    for word in words:
        if word in word_vectors:
            vector += word_vectors[word]
    return vector

def train():

    df_train = load_data('../data/train.json')
    batch_size = df_train.shape[0]
    df_dev = load_data('../data/dev.json')

    sentences = df_train['text'].tolist() + df_dev['text'].tolist()
    # train_word2vec(sentences)
    word_vectors = load_word2vec().wv
    vector_size = word_vectors.vector_size


    embeddings_train = df_train['text'].apply(lambda x: to_vector(x, word_vectors, vector_size)).tolist()
    embeddings_dev = df_dev['text'].apply(lambda x: to_vector(x, word_vectors, vector_size)).tolist()

    X_train = embeddings_train
    y_train = df_train.drop(columns=['text']).values


    X_dev = embeddings_dev
    y_dev = df_dev.drop(columns=['text']).values

    clf = MultiOutputClassifier(svm.SVC(decision_function_shape='ovo',probability=True))

    max_accuracy = 0


    indices = np.random.choice(len(X_train), size=batch_size, replace=False)
    X_train_batch, y_train_batch = np.array(X_train)[indices], y_train[indices]

    clf.fit(X_train_batch, y_train_batch)

    predictions = clf.predict(X_dev)
    accuracies = []
    for acc in range(y_dev.shape[1]):  # 遍历每个输出
        accuracies.append(accuracy_score(y_dev[:, acc], predictions[:, acc]))

    accuracy = np.mean(accuracies)  # 计算平均准确率
    print(f"accuracy: {accuracy * 100:.2f}%")

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        joblib.dump(clf, 'best_model.pkl')

def test():
    df_test = load_data('../data/test.json')

    word_vectors = load_word2vec().wv
    vector_size = word_vectors.vector_size
    embeddings = df_test['text'].apply(lambda x: to_vector(x, word_vectors, vector_size)).tolist()

    X = embeddings
    y = df_test.drop(columns=['text']).values

    model = joblib.load('./best_model.pkl')

    y_pred = model.predict(X)

    y_pred_proba = model.predict_proba(X)


    # 计算准确率
    accuracy = calculate_accuracy(y, y_pred)
    print(f"Accuracy Score: {accuracy}")

    # 计算AUC
    auc = calculate_auc(y, y_pred_proba)
    print(f"AUC Score: {auc}")

    # 计算F1得分
    f1 = calculate_f1(y, y_pred)
    print(f"F1 Score: {f1}")

def main():
    train()
    test()

if __name__ == "__main__":
    main()
