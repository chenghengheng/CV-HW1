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

    ip = np.append(X, 1).dot(inputW)
    fp = np.tanh(ip)  # 激活函数
    yhat = np.append(fp, 1).dot(outputW)

    Err = yhat - y
    f = sum(Err ** 2)  # loss：平方损失
    error = 2 * Err  # 误差

    gOutput = np.append(fp, 1).reshape(-1, 1).dot(error.reshape(1, -1))  # 反向传播计算梯度
    gw[nVars * nHidden[0]:] = gOutput[:gOutput.shape[0] - 1, :].reshape(-1, 1)
    gb[nHidden[0]:] = gOutput[gOutput.shape[0] - 1, :].reshape(-1, 1)

    gInput = np.append(X, 1).reshape(-1, 1).dot((np.dot(error, outputW[:outputW.shape[0] - 1, :].T) * (1 - np.tanh(ip) ** 2)).reshape(1, -1))
    gw[:nVars * nHidden[0]] = gInput[:gInput.shape[0] - 1, :].reshape(-1, 1)
    gb[:nHidden[0]] = gInput[gInput.shape[0] - 1, :].reshape(-1, 1)

    fp = np.append(fp, 1)

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

beta = 0.9
maxIter = 100000
stepSize = [1e-4, 0.8 * 1e-4, 0.6 * 1e-4, 0.4 * 1e-4]  # 学习率下降

nPara = d * nHidden[0] + nHidden[0] * nLabels
nBias = nHidden[0] + nLabels
w = np.random.normal(0, 1, (nPara, 1))
wpre = np.zeros((nPara, 1))
b = np.random.normal(0, 1, (nBias, 1))

los = []
train = []
test = []

for i in range(maxIter):
    if not i % 5000:
        yhat = predict(w, b, X, nHidden, nLabels)
        ytest = predict(w, b, Xtest, nHidden, nLabels)
        tre = sum(yhat != y_train) / n
        tee = sum(ytest != y_test) / t
        train.append(tre)
        test.append(tee)
        print("Train iteration = %d, training error = %f, test error = %f" % (i, tre, tee))
    j = math.ceil(random.random() * n - 1)
    f, gw, gb, _ = loss(w, b, X[j], yExpanded[j], nHidden, nLabels)
    los.append(f)

    temp = w  # SGD
    w = w - stepSize[i // 25000] * (gw + 0.1 * w) + beta * (w - wpre)  # 为方便计算，将L2正则化直接作用在梯度上，同时使用momentum
    wpre = temp
    b = b - stepSize[i // 25000] * gb



yhat1 = predict(w, b, X, nHidden, nLabels)
tre = sum(yhat1 != y_train) / n
train.append(tre)
print("\nTraining error with final model = %f" % tre)
yhat2 = predict(w, b, Xtest, nHidden, nLabels)
tee = sum(yhat2 != y_test) / t
test.append(tee)
print("Test error with final model = %f" % tee)

np.savetxt("weights.txt", w)  # 保存模型
np.savetxt("bias.txt", b)
