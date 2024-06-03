from .math import *
from enum import Enum

class ActivationFn(Enum):
    SIGMOID = sigmoid
    RELU = relu
    LEAKY_RELU = leaky_relu

class Neuron():
    def __init__(self, b=0, activation_fn=ActivationFn.SIGMOID) -> None:
        self.bias = b
        self.value = 0
        self.activation_fn=activation_fn