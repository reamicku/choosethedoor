from math import sqrt
import numpy as np

def relu(x):
    return np.maximum(0.0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return exp_x / np.sum(exp_x)