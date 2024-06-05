from math import sqrt
import numpy as np

# Activation functions

def identity(x):
    return x


def binary(x):
    if x >= 0.0: return 1.0
    return 0.0


def tanh(x):
    return np.tanh(x)


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

# Random distribution functions


def linear_distrib(size: int, lower: float, upper: float) -> np.array:
    return lower + np.random.rand(size) * (upper - lower)


def xavier_distrib(n_input: int, size: int, lower: float, upper: float) -> np.array:
    lowerV, upperV = lower / sqrt(n_input), upper / sqrt(n_input)
    return lowerV + np.random.rand(size) * (upperV - lowerV)


def he_distrib(size: int, lower: float, upper: float) -> np.array:
    stddev = sqrt(2.0 / size)
    return np.random.normal(0, stddev, size) * (upper - lower)
