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


def print_image(data):
    plt.imshow(data.reshape((28, 28)), cmap = 'gray_r')
    plt.show()


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
    plt.show()
    plt.savefig('Histogram-counts.png')


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
            y_hat = test_half_Y[i]
            if (y_hat == y_predict):
                acc_cnt += 1
            confusion_matrix[y_hat][y_predict] += 1
    if (not isTest):
        np.save('confusion.npy', confusion_matrix)
        print('accuracy = %.3f' % (acc_cnt / test_half_X.shape[0]))
    else:
        with open('submission.csv', 'w') as fw:
            fw.write('ImageId,Label\n')
            for i in range(len(predictions)):
                fw.write('%d,%d\n'%(i, int(predictions[i])))


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    trainY, trainX = read(train_path)
    testY, testX = read(test_path)
    print_image(trainX[0])
    #histogram(trainY)
    #find_nearest(trainX, trainY)
    #_a, _b, _c, _d = split_data(trainX, trainY)
    #train_and_test(_a, _b, _c, _d)
    train_and_test(testX, testY, trainX, trainY, True)
    matrix = np.load('confusion.npy').astype(int)
    print(matrix)


