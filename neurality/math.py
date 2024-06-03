import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return max(0.0, x)


def leaky_relu(x):
    if x > 0:
        return x
    else:
        return 0.01*x


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
