import sklearn
import numpy as np
import os
import string
import nltk
from nltk.stem import WordNetLemmatizer as wnl
from nltk import corpus
import json

def preprocess(text, stopwords):
    lemmatizer = wnl()
    text = text.lower().translate(None, string.punctuation)
    items = text.split(' ')
    ret = []
    for i, item in enumerate(items):
        if (not item in stopwords) and (len(item) > 0):
            ret.append(lemmatizer.lemmatize(item.decode('utf-8')).encode('utf-8'))
    return ret

def scan_and_preprocess():
    files = os.listdir('data/sentiment labelled sentences')
    nltk.download('wordnet')
    nltk.download('stopwords')
    stopwords = set(corpus.stopwords.words('english'))
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for eachFile in files:
        if eachFile.endswith('labelled.txt'):
            with open(os.path.join('data/sentiment labelled sentences', eachFile), 'r') as fr:
                texts = []
                labels = []
                for eachLine in fr:
                    items = eachLine.rstrip('\n').split('\t')
                    texts.append(preprocess(items[0], stopwords))
                    labels.append(int(items[1]))
                cnt0 = 0
                cnt1 = 0
                for i in range(len(texts)):
                    if (labels[i] == 0):
                        cnt0 += 1
                        if (cnt0 <= 400):
                            train_texts.append(texts[i])
                            train_labels.append(labels[i])
                        else:
                            test_texts.append(texts[i])
                            test_labels.append(labels[i])
                    else:
                        cnt1 += 1
                        if (cnt1 <= 400):
                            train_texts.append(texts[i])
                            train_labels.append(labels[i])
                        else:
                            test_texts.append(texts[i])
                            test_labels.append(labels[i])

                labels = np.array(labels, dtype=int)
                print (eachFile, (labels == 0).sum(-1), (labels == 1).sum(-1))
    with open('data/sentiment labelled sentences/train.json', 'w') as fw:
        data = {'texts' : train_texts, 'labels' : train_labels}
        json.dump(data, fw)
    with open('data/sentiment labelled sentences/test.json', 'w') as fw:
        data = {'texts' : test_texts, 'labels' : test_labels}
        json.dump(data, fw)
    return {'texts' : train_texts, 'labels' : train_labels}, {'texts' : test_texts, 'labels' : test_labels}


def build_bow(train, test, ngram = 1):
    def sparse(x):
        ret = []
        for index, item in enumerate(x):
            if item != 0:
               ret.append((index, item))
        return ret
    #train = None
    #test = None
    #with open('data/sentiment labelled sentences/train.json', 'r') as fr:
    #    train = json.load(fr)
    #with open('data/sentiment labelled sentences/test.json', 'r') as fr:
    #    test = json.load(fr)
    from sklearn.preprocessing import StandardScaler
    train_texts = train['texts']
    test_texts = test['texts']
    total_count = []
    pos = {}
    for text in train_texts:
        for i in range(len(text) - ngram + 1):
            item = []
            for j in range(ngram):
                item.append(text[i+j])
            item = tuple(item)

            if not (item in pos):
                pos[item] = len(total_count)
                total_count.append(item)
            else:
                total_count[pos[item]] = item


    train_features = []
    for text in train_texts:
        feature = [0 for i in range(len(total_count))]
        for i in range(len(text) - ngram + 1):
            item = []
            for j in range(ngram):
                item.append(text[i + j])
            item = tuple(item)

            feature[pos[item]] += 1
        train_features.append(feature)

    train_features = np.array(train_features, dtype=float)
    print sparse(train_features[1234])
    print train_texts[1234]
    print sparse(train_features[2134])
    print train_texts[2134]

    scaler = StandardScaler()
    scaler.fit(train_features)
    mean_train = np.mean(train_features, axis=0)
    std_train = np.std(train_features, axis=0)
    #train_features = (train_features - mean_train) / std_train
    #train_features = scaler.transform(train_features)
    #train_features = train_features / np.maximum(np.sum(train_features, 1), 1).reshape(-1, 1)
    train_features = np.log(train_features+1)
    test_features = []
    for text in test_texts:
        feature = [0 for i in range(len(total_count))]
        for i in range(len(text) - ngram + 1):
            item = []
            for j in range(ngram):
                item.append(text[i + j])
            item = tuple(item)
            if (item in pos):
                feature[pos[item]] += 1
        test_features.append(feature)

    test_features = np.array(test_features, dtype=float)
    #test_features = test_features / np.maximum(np.sum(test_features, 1), 1).reshape(-1, 1)
    #test_features = (test_features - mean_train) / std_train
    #test_features = scaler.transform(test_features)
    test_features = np.log(test_features+1)

    print(len(total_count))
    return train_features, np.array(train['labels'], dtype=int) , test_features, np.array(test['labels'], dtype=int), np.array(total_count)


def log_gaussian(x, mean, std):
    exponent = -0.5 * np.sum((x - mean) ** 2 / std, 1)
    return -0.5 * np.sum(np.log(2 * np.pi * std)) + exponent

def log_gaussian_x(x, mean, std):
    import math
    exponent = -0.5 * ((x - mean) ** 2) / std
    return -0.5 * math.log(2 * math.pi * std) + exponent


def naive_bayes(train_features, train_labels, test_features, test_labels):
    mean_0 = train_features[np.where(train_labels==0)].mean(axis=0)
    std_0 = train_features[np.where(train_labels==0)].var(axis=0) + 1e-9 * np.var(train_features, 0).max()
    mean_1 = train_features[np.where(train_labels == 1)].mean(axis=0)
    std_1 = train_features[np.where(train_labels == 1)].var(axis=0) + 1e-9 * np.var(train_features, 0).max()

    log_likelyhood0 = log_gaussian(test_features, mean_0, std_0)
    log_likelyhood1 = log_gaussian(test_features, mean_1, std_1)
    pred0 = np.where(log_likelyhood0 > log_likelyhood1)
    pred1 = np.where(log_likelyhood0 <= log_likelyhood1)


    acc = (test_labels[pred0].shape[0] - test_labels[pred0].sum() + test_labels[pred1].sum()) * 1.0 / test_labels.shape[0]

    confusion_matrix = np.zeros(shape=(2, 2))
    confusion_matrix[0][0] = np.count_nonzero(test_labels[pred0] == 0)
    confusion_matrix[0][1] = np.count_nonzero(test_labels[pred0] == 1)
    confusion_matrix[1][1] = np.count_nonzero(test_labels[pred1] == 1)
    confusion_matrix[1][0] = np.count_nonzero(test_labels[pred1] == 0)
    print(acc)
    print(confusion_matrix)

def logistic_regression(train_features, train_labels, test_features, test_labels):
    from sklearn.linear_model import LogisticRegressionCV
    clf1 = LogisticRegressionCV(penalty = 'l1', cv = 5, solver='liblinear', multi_class='ovr', max_iter=1000).fit(train_features, train_labels)
    clf2 = LogisticRegressionCV(penalty = 'l2', cv = 5, solver='liblinear', multi_class='ovr', max_iter=1000).fit(train_features, train_labels)
    return clf1, clf2

def analyze(clf1, clf2, word_dict):
    coef_abs = -np.fabs(clf1.coef_)
    print(word_dict[np.argsort(coef_abs)[:,0:20]])
    print(clf1.coef_.reshape(-1, )[np.argsort(coef_abs)[:,0:20]])
    coef_abs = -np.fabs(clf2.coef_)
    print(word_dict[np.argsort(coef_abs)[:,0:20]])
    print(clf2.coef_.reshape(-1, )[np.argsort(coef_abs)[:,0:20]])


if __name__ == '__main__':
    train, test = scan_and_preprocess()
    train_features, train_labels, test_features, test_labels, word_dict = build_bow(train, test, ngram = 1)
    naive_bayes(train_features, train_labels, test_features, test_labels)
    clf1, clf2 = logistic_regression(train_features, train_labels, test_features, test_labels)
    analyze(clf1, clf2, word_dict)

    train_features, train_labels, test_features, test_labels, word_dict = build_bow(train, test, ngram=2)
    naive_bayes(train_features, train_labels, test_features, test_labels)
    clf1, clf2 = logistic_regression(train_features, train_labels, test_features, test_labels)
    analyze(clf1, clf2, word_dict)