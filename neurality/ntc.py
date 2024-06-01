from .neuron import Neuron
from .connection import Connection

class NTC():
    def __init__(self, neuron, connection, w=0) -> None:
        self.neuron = neuron
        self.connection = connection
        self.weight = w
        self.value = 0