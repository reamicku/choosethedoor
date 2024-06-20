"""
This module provides a collection of mathematical functions commonly
used inneural networks and machine learning.

All functions in this module are designed to work with NumPy arrays (`np.ndarray`)
and can handle arrays of any shape, making them suitable for batched operations
where multiple instances are processed simultaneously.
"""
import numpy as np

def relu(x):
    """
    Compute the Rectified Linear Unit (ReLU) for the elements of an input array.
    
    The ReLU function is defined as:
        f(x) = 0   if x < 0
        f(x) = x   if x >= 0
    
    Parameters:
    x (np.ndarray): An input array of any shape.
    
    Returns:
    np.ndarray: An array the same shape as 'x' with the ReLU function
                applied to each element.
    """
    return np.maximum(0.0, x)

def leaky_relu(x):
    """
    Compute the Leaky Rectified Linear Unit (Leaky ReLU) for the elements of an input array.
    
    The Leaky ReLU function allows a small, non-zero gradient when the unit is not active,
    mitigating the "dying ReLU" problem. It is defined as:
        f(x) = alpha * x   if x < 0
        f(x) = x            if x > 0
    
    In this implementation, `alpha` is set to 0.01.
    
    Parameters:
    x (np.ndarray): An input array of any shape.
    
    Returns:
    np.ndarray: An array the same shape as 'x' with the Leaky ReLU function
                applied to each element.
    """
    return np.maximum(0.01*x, x)

def tanh(x):
    """
    Compute the hyperbolic tangent (tanh) for the elements of an input array.
    
    The tanh function is a rescaled version of the logistic sigmoid function.
    It maps the input values from the range (-∞, +∞) to the range (-1, 1).
    It is defined as:
        f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    This function is similar to the sigmoid function but has the advantage of
    being zero-centered, which can lead to faster convergence of gradients in
    optimization algorithms.
    
    Parameters:
    x (np.ndarray): An input array of any shape.
    
    Returns:
    np.ndarray: An array the same shape as 'x' with the tanh function
                applied to each element.
    """
    return np.tanh(x)

def softmax(x):
    """
    Compute the softmax function for the elements of an input vector x.
    
    The softmax function converts an arbitrary vector of real-valued elements
    to a vector of probabilities, where each element is the probability that
    a given sample belongs to a particular class.
    
    The softmax function is defined as:
        f(xi) = exp(xi) / sum(exp(x))
    
    where xi is an element of the input vector x.
    
    Parameters:
    x (np.ndarray): An input array of any shape representing logits or unnormalized
                  log-probabilities.
    
    Returns:
    np.ndarray: An array the same shape as 'x' with the softmax function
                applied to transform the elements into probabilities.
    """
    exp_x = np.exp(x - np.max(x))  # Subtracting np.max(x) for numerical stability
    return exp_x / np.sum(exp_x)
