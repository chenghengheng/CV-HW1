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

    y = np.argmax(yhat, axis=1)

    return y


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

n, d = x_train.shape[0], x_train.shape[1]
d = d * d
x_train = x_train.reshape(n, d)

t, d1 = x_test.shape[0], x_test.shape[1]
d1 = d1 * d1
x_test = x_test.reshape(t, d1)

nLabels = max(y_train) + 1
yExpanded = binary(y_train, nLabels)

X, mu, sigma = standardize(x_train)
Xtest = standardize_(x_test, mu, sigma)

nHidden = [200]

w = np.loadtxt("weights.txt")
b = np.loadtxt("bias.txt")

yhat1 = predict(w, b, X, nHidden, nLabels)
print("Training error with final model = %f" % (sum(yhat1 != y_train) / n))
yhat2 = predict(w, b, Xtest, nHidden, nLabels)
print("Test error with final model = %f" % (sum(yhat2 != y_test) / t))
