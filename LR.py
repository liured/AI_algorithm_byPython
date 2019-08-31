import numpy as np


def Sigmoid(X):
    return 1.0 / (1+np.exp(-X))

def trainLR(train_x, train_y, args):
    numSample, numFeat = np.shape(train_x)
    alpha = args['alpha']
    maxIter = args['maxIter']
    W = np.ones((numFeat, 1))  # #设置初始的参数，默认值为1

    for k in range(maxIter):
        if args['optimize'] == 'GD':
            Y_pred = Sigmoid(train_x * W)
            error = train_y - Y_pred
            W = W + alpha * train_x.transpose() * error
        elif args['optimize'] == 'SGD':
            for i in range(numSample):
                Y_pred = Sigmoid(train_x[i,:] * W)
                error = train_y[i, 0] - Y_pred
                W = W + alpha * train_x[i,:].transpose() * error
    return W

def testLR(test_x, test_y, W):
    numSample, numFeat = np.shape(test_x)
    matchCount = 0
    for i in range(numSample):
        predict = Sigmoid(test_x[i,:] * W)[0,0] > 0.5
        if predict == bool(test_y[i,0]):
            matchCount += 1
    acc = float(matchCount) / numSample

    return acc

import matplotlib.pyplot as plt
def showLogRegres(weights, train_x, train_y):
    # 画点
    numSamples, numFeatures = np.shape(train_x)
    if numFeatures != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1
    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

     # 画线
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

if __name__ == "__main__":
    train_x = []
    train_y = []
    f = open('utils/data.txt')
    for line in f.readlines():
        line = line.strip().split()
        train_x.append([1.0, float(line[0]), float(line[1])])
        train_y.append(float(line[2]))
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).transpose()
    # print(train_x)
    # print(train_y)
    args = {'alpha': 0.01,'maxIter':200,'optimize':'GD'}
    weights = trainLR(train_x, train_y,args)
    print(weights)
    acc = testLR(train_x, train_y, weights)

    print(acc)
    showLogRegres(weights, train_x,train_y)
