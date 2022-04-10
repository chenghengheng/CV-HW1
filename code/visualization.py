import matplotlib.pyplot as plt
import numpy as np

w = np.loadtxt("weights.txt")
w1 = w[:78400].reshape(100, 28, 28)
w2 = w[78400:].reshape(10, 10, 10)


plt.figure(figsize=(10, 10))
for x in range(100):
    plt.subplot(10, 10, x + 1)
    plt.imshow(w1[x, :, :])

plt.figure(figsize=(2, 5))
for x in range(10):
    plt.subplot(2, 5, x + 1)
    plt.imshow(w2[x, :, :])

