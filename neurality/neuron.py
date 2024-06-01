class Neuron:
    def __init__(self, b=0) -> None:
        self.bias = b
        self.value = 0

class InputNeuron(Neuron):
    def __init__(self) -> None:
        super(Neuron, self).__init__()
        pass

class OutputNeuron(Neuron):
    def __init__(self) -> None:
        super(Neuron, self).__init__()
        pass