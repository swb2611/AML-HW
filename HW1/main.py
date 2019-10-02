import csv
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm



def read(data_path):
    images = []
    labels = []
    with open(data_path) as fr:
        f_csv = csv.reader(fr)
        cnt = 0
        for line in f_csv:
            if (cnt > 0):
                if (len(line) == 785):
                    images.append(line[1:])
                    labels.append(line[0])
                else:
                    images.append(line)
            else:
                cnt += 1

    if (len(labels) == 0):
        return None, np.array(images, dtype = float)
    else:
        return np.array(labels, dtype = int), np.array(images, dtype = float)


def print_image(data, name = None):
    plt.imshow(data.reshape((28, 28)), cmap = 'gray_r')
    if name:
        plt.savefig('images/%s' % name)
    plt.show()

def display_each(dataX, dataY):
    for i in range(10):
        for j in range(dataX.shape[0]):
            if (dataY[j] == i):
                print_image(dataX[j], name = 'digit%d.png'%i)
                break


def histogram(data):
    counter = np.array([0 for i in range(10)])
    for label in data:
        counter[label] +=1
    dis = normalize(counter.reshape(-1, 1), axis=0, norm='max')
    fig = plt.figure()
    x = [i for i in range(10)]
    plt.bar(x, dis.reshape(-1), 0.2, color = 'green')
    plt.xticks(np.arange(0, 10, 1))
    plt.xlabel('Digit')
    plt.ylabel('Normalized Counts')
    plt.savefig('Histogram-counts.png')
    plt.show()


def dist(u, v):
    return np.sqrt(np.sum(np.square(u - v), axis=-1))


def find_nearest(X, Y):
    n = X.shape[0]
    cur = 0
    for i in range(10):
        for j in range(n):
            if (Y[j] == i):
                pick = (1e9, -1)
                for k in range(n):
                    if (k != j):
                        pick = min(pick, (dist(X[j], X[k]), k))
                print(j, pick[1], Y[j], Y[pick[1]])
                break


def knn(u, train_x, train_y, k):
    distance = dist(train_x, u)
    indices = np.argpartition(distance, k)[:k]
    counts = np.bincount(train_y[indices])
    return np.argmax(counts)


def split_data(trainX, trainY):
    n = trainX.shape[0]
    ind = np.random.choice(n, int(n/2), replace=False)
    train_half_X = trainX[ind]
    train_half_Y = trainY[ind]

    rest = n - ind.shape[0]
    ind_ = []
    mark = np.zeros(shape=(n, ))
    for i in ind:
        mark[i] = 1
    for i in range(n):
        if (mark[i] == 0):
            ind_.append(i)
    ind_ = np.array(ind_)

    test_half_X = trainX[ind_]
    test_half_Y = trainY[ind_]

    #print(test_half_Y[0], knn(test_half_X[0], train_half_X, train_half_Y, 10))
    return test_half_X, test_half_Y, train_half_X, train_half_Y


def train_and_test(test_half_X, test_half_Y, train_half_X, train_half_Y, isTest = False):
    acc_cnt = 0
    confusion_matrix = np.zeros(shape=(10, 10))
    predictions = []
    for i in tqdm(range(test_half_X.shape[0])):
        x = test_half_X[i]
        y_predict = knn(x, train_half_X, train_half_Y, 25)
        predictions.append(y_predict)
        if (not isTest):
            y_fact = test_half_Y[i]
            if (y_fact == y_predict):
                acc_cnt += 1
            confusion_matrix[y_fact][y_predict] += 1
    if (not isTest):
        np.save('confusion.npy', confusion_matrix)
        print('accuracy = %.3f' % (acc_cnt / test_half_X.shape[0]))
    else:
        with open('submission.csv', 'w') as fw:
            fw.write('ImageId,Label\n')
            for i in range(len(predictions)):
                fw.write('%d,%d\n'%(i+1, int(predictions[i])))

def binary_classify(dataX, dataY):
    import os.path
    from sklearn.metrics import auc
    from pathlib import Path
    npfile = Path('genuine.npy')
    genuine = None
    impostor = None
    if not npfile.exists():
        ind = []
        for i in range(dataY.shape[0]):
            if (dataY[i] == 0 or dataY[i] == 1):
                ind.append(i)

        X = dataX[ind]
        Y = dataY[ind]

        n = len(ind)

        genuine = []
        impostor = []
        print(n)
        for i in tqdm(range(n)):
            for j in range(i+1, n):
                if (Y[i] == Y[j]):
                    genuine.append(dist(X[i], X[j]))
                else:
                    impostor.append(dist(X[i], X[j]))

        genuine = np.array(genuine, dtype=float)
        impostor = np.array(impostor, dtype=float)
        np.save('genuine.npy', genuine)
        np.save('impostor.npy', impostor)
    else:
        genuine = np.load('genuine.npy')
        impostor = np.load('impostor.npy')

    num_bins = 20
    n, bins, patches = plt.hist(genuine, num_bins, facecolor='blue', alpha=0.5, label='genuine', density = True)
    plt.hist(impostor, num_bins, facecolor='yellow', alpha=0.5, label='impostor', density = True)
    plt.legend(loc='upper left', prop={'size': 15});
    plt.title('Histogram of genuine and impostor distances')
    plt.savefig('histogram-binary')

    plt.show()
    n_genuine = genuine.shape[0]
    n_impostor = impostor.shape[0]
    alldistances = np.concatenate([genuine, impostor])

    args = np.argsort(alldistances)
    tp = 0
    fp = 0

    x = []
    y = []
    for i in range(args.shape[0]):
        if (args[i] < n_genuine):
            tp += 1
        else:
            fp += 1
        tpr = tp / n_genuine
        fpr = fp / n_impostor
        x.append(fpr)
        y.append(tpr)

    x = np.array(x, dtype = float)
    y = np.array(y, dtype = float)

    plt.plot(x, y)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.savefig('images/roc_curve.png')
    plt.show()
    roc_auc = auc(x, y)
    print(roc_auc)
def print_confusion():
    matrix = np.load('confusion.npy').astype(int)


    hline = '\\hline'
    print(hline)
    s = "\\backslashbox{digit-fact}{digit-predict}"
    for i in range(10):
        s = s + ' & '
        s = s + "%d"%i
    s = s + '\\\\'

    print(s)
    print(hline)
    matrix = matrix / np.sum(matrix, axis = -1)
    for i in range(matrix.shape[0]):
        s = "%d"%i
        for j in range(matrix.shape[1]):
            s = s + ' & '
            s = s + '%.3f'%(matrix[i][j])
        s = s + '\\\\'
        print(s)
        print(hline)


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    trainY, trainX = read(train_path)
    testY, testX = read(test_path)
    display_each(trainX, trainY)
    histogram(trainY)
    find_nearest(trainX, trainY)
    binary_classify(trainX, trainY)
    _a, _b, _c, _d = split_data(trainX, trainY)
    train_and_test(_a, _b, _c, _d)
    train_and_test(testX, testY, trainX, trainY, True)


