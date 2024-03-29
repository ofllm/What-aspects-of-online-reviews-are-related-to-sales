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

model_path = '../../model/fasttext/cc.zh.300.bin'
def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, path='./word2vec.model'):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.save(path)

def load_word2vec(path='./word2vec.model'):
    return load_facebook_model(model_path)

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

    clf = MultiOutputClassifier(svm.SVC(decision_function_shape='ovo'))

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

    accuracies = []
    for i in range(y.shape[1]):  # 遍历每个输出
        accuracies.append(accuracy_score(y[:, i], y_pred[:, i]))

    accuracy = np.mean(accuracies)  # 计算平均准确率

    print(f'The accuracy of the model on the test set is: {accuracy * 100:.2f}%')

def main():
    train()
    test()

if __name__ == "__main__":
    main()
