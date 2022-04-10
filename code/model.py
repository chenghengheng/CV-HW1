import tensorflow as tf
from para2 import *
import numpy as np
import random
import math

np.random.seed(1)
random.seed(1)

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
ytExpanded = binary(y_test, nLabels)

X, mu, sigma = standardize(x_train)
Xtest = standardize_(x_test, mu, sigma)

nHidden = [100]

w = np.loadtxt("weights.txt")
b = np.loadtxt("bias.txt")

yhat1 = predict(w, b, X, nHidden, nLabels)
print("Training error with final model = %f" % (sum(yhat1 != y_train) / n))
yhat2 = predict(w, b, Xtest, nHidden, nLabels)
print("Test error with final model = %f" % (sum(yhat2 != y_test) / t))
print("\nTest accuracy with final model = %f" % 1 - (sum(yhat2 != y_test) / t))
