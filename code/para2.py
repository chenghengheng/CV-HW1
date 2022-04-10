import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math


def binary(label, nLabels):
    """
    将数据标签转化为二值矩阵
    :param label: 数据标签 y
    :param nLabels: 类别数量
    :return: y的扩展矩阵
    """
    n = len(label)
    op = -np.zeros((n, nLabels))
    for i in range(n):
        op[i][label[i]] = 1
    return op


def standardize(X):
    """
    数据标准化
    :param X:原始数据
    :return: 标准化后的数据，均值，标准差
    """
    mu = np.average(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma < 1e-10] = 1
    S = X - mu
    S = S / sigma
    return S, mu, sigma


def standardize_(X, mu, sigma):
    """
    数据标准化
    :param X:原始数据
    :param mu: 均值
    :param sigma: 方差
    :return: 标准化后的数据
    """
    S = X - mu
    S = S / sigma
    return S


def loss(w, b, X, y, nHidden, nLabels):
    nInstances, nVars = (X.shape[0], X.shape[1]) if len(X.shape) > 1 else (1, X.shape[0])

    inputW = np.zeros((nVars + 1, nHidden[0]))
    inputW[:nVars, :] = w[:nVars * nHidden[0]].reshape(nVars, nHidden[0])
    inputW[nVars, :] = b[:nHidden[0]].reshape(1, nHidden[0])

    outputW = np.zeros((nHidden[0] + 1, nLabels))
    outputW[:nHidden[0], :] = w[nVars * nHidden[0]:].reshape(nHidden[0], nLabels)
    outputW[nHidden[0], :] = b[nHidden[0]:].reshape(1, nLabels)

    gw = np.zeros(w.shape)
    gb = np.zeros(b.shape)

    if nInstances == 1:
        ip = np.append(X, 1).dot(inputW)
        fp = np.tanh(ip)  # 激活函数
        yhat = np.append(fp, 1).dot(outputW)

        py = np.exp(yhat) / np.sum(np.exp(yhat))
        index = np.argmax(py)

        f = 0
        for i in range(nInstances):
            f -= math.log(py[index])  # loss：交叉熵

        error = py - (y == 1)  # 误差

        gOutput = np.append(fp, 1).reshape(-1, 1).dot(error.reshape(1, -1))  # 反向传播计算梯度
        gw[nVars * nHidden[0]:] = gOutput[:gOutput.shape[0] - 1, :].reshape(-1, 1)
        gb[nHidden[0]:] = gOutput[gOutput.shape[0] - 1, :].reshape(-1, 1)

        gInput = np.append(X, 1).reshape(-1, 1).dot(
            (np.dot(error, outputW[:outputW.shape[0] - 1, :].T) * (1 - np.tanh(ip) ** 2)).reshape(1, -1))
        gw[:nVars * nHidden[0]] = gInput[:gInput.shape[0] - 1, :].reshape(-1, 1)
        gb[:nHidden[0]] = gInput[gInput.shape[0] - 1, :].reshape(-1, 1)

        fp = np.append(fp, 1)

        return f, gw, gb, fp

    one = np.ones((nInstances, 1))

    ip = np.c_[X, one].dot(inputW)
    fp = np.tanh(ip)  # 激活函数
    yhat = np.c_[fp, one].dot(outputW)

    py = np.exp(yhat)
    for j in range(nInstances):
        py[j] = py[j] / py[j].sum()

    f = 0
    for i in range(nInstances):
        f -= np.log(py[i, y[i]])  # 交叉熵

    return f, gw, gb, fp


def predict(w, b, X, nHidden, nLabels):
    nInstances, nVars = (X.shape[0], X.shape[1]) if len(X.shape) > 1 else (1, X.shape[0])

    inputW = np.zeros((nVars + 1, nHidden[0]))
    inputW[:nVars, :] = w[:nVars * nHidden[0]].reshape(nVars, nHidden[0])
    inputW[nVars, :] = b[:nHidden[0]].reshape(1, nHidden[0])

    outputW = np.zeros((nHidden[0] + 1, nLabels))
    outputW[:nHidden[0], :] = w[nVars * nHidden[0]:].reshape(nHidden[0], nLabels)
    outputW[nHidden[0], :] = b[nHidden[0]:].reshape(1, nLabels)

    one = np.ones((nInstances, 1))
    ip = np.c_[X, one].dot(inputW)
    fp = np.tanh(ip)  # 激活函数
    yhat = np.c_[fp, one].dot(outputW)

    py = np.exp(yhat) / np.sum(np.exp(yhat))
    y = np.argmax(py, axis=1)

    return y


def para(stepSize, nHidden, Lambda, X, y_train, n, d, Xtest, y_test, t, nLabels, yExpanded, index):
    np.random.seed(1)
    random.seed(1)
    
    beta = 0.9
    maxIter = 100000

    nPara = d * nHidden[0] + nHidden[0] * nLabels
    nBias = nHidden[0] + nLabels
    w = np.random.normal(0, 1, (nPara, 1))
    wpre = np.zeros((nPara, 1))
    b = np.random.normal(0, 1, (nBias, 1))

    loss_train = []
    loss_test = []
    accu_test = []

    for i in range(maxIter):
        if not i % 5000:
            yhat = predict(w, b, X, nHidden, nLabels)
            ytest = predict(w, b, Xtest, nHidden, nLabels)

            if i:
                loss1 = loss(w, b, X, y_train, nHidden, nLabels)[0] / 60000
                loss2 = loss(w, b, Xtest, y_test, nHidden, nLabels)[0] / 10000
                loss_train.append(loss1)
                loss_test.append(loss2)

            tee = sum(ytest != y_test) / t
            accu_test.append(tee)
            print("Train iteration = %d, training error = %f, test error = %f" % (i, sum(yhat != y_train) / n, tee))
        j = math.ceil(random.random() * n - 1)
        _, gw, gb, _ = loss(w, b, X[j], yExpanded[j], nHidden, nLabels)

        step = stepSize * np.cos((i / maxIter) * np.pi / 2)  # 学习率下降

        temp = w  # SGD
        w = w - step * (gw + Lambda * w) + beta * (w - wpre)  # 为方便计算，将L2正则化直接作用在梯度上，同时使用momentum
        wpre = temp
        b = b - step * gb

    yhat1 = predict(w, b, X, nHidden, nLabels)
    print("Training error with final model = %f" % (sum(yhat1 != y_train) / n))
    yhat2 = predict(w, b, Xtest, nHidden, nLabels)
    tee = sum(yhat2 != y_test) / t
    accu_test.append(tee)
    print("Test error with final model = %f" % tee)

    np.savetxt("weights_%d hidden_%f step_%f lambda.txt" % (nHidden[0], stepSize, Lambda), w)  # 保存模型
    np.savetxt("bias_%d hidden_%f step_%f lambda.txt" % (nHidden[0], stepSize, Lambda), b)

    return loss_train, loss_test, accu_test
