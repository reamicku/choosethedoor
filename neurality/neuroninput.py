from .neuron import Neuron

class NeuronConnection():
    def __init__(self, incomingNeuron: Neuron, parentNeuron: Neuron, w: int = 0) -> None:
        self.incomingNeuron = incomingNeuron
        self.parentNeuron = parentNeuron
        self.weight = w
        self.value = 0
