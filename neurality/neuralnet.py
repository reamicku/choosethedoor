from .neuron import Neuron, ActivationFn
from .neuroninput import NeuronConnection
from .math import softmax
from .utils import generate_pairs, generate_non_overlapping_pairs
import random
import numpy as np


class NeuralNet():
    def __init__(self, inputNeuronCount: int, outputNeuronCount: int, neuronInitialCount: int = 64, connectionInitialCount: int = 196) -> None:
        assert inputNeuronCount > 0, "inputNeuronCount cannot be less than 1"
        assert outputNeuronCount > 0, "outputNeuronCount cannot be less than 1"
        assert neuronInitialCount > 0, "neuronInitialCount cannot be less than 1"
        self.neuronList = []
        self.neuronInputList = []
        self.inputNeuronIDs = []
        self.outputNeuronIDs = []
        self.totalNeuronCount = inputNeuronCount + \
            outputNeuronCount + neuronInitialCount
        self.inputNeuronValues = []

        selectedIDs = []
        i = 0
        while True:
            randId = random.randrange(0, self.totalNeuronCount-1)
            if randId not in selectedIDs:
                self.inputNeuronIDs.append(randId)
                selectedIDs.append(randId)
                i += 1
            if i >= inputNeuronCount:
                break
        i = 0
        while True:
            randId = random.randrange(0, self.totalNeuronCount-1)
            if randId not in selectedIDs:
                self.outputNeuronIDs.append(randId)
                selectedIDs.append(randId)
                i += 1
            if i >= outputNeuronCount:
                break

        self.initializeNeurons(
            inputNeuronCount + outputNeuronCount + neuronInitialCount)
        self.initializeNeuronConnections(connectionInitialCount)

        dummyInput = []
        for i in range(0, inputNeuronCount):
            dummyInput.append(0)
        self.setInputNeuronValues(dummyInput)

    def __str__(self) -> str:
        return f"NeuralNet(size={len(self.neuronList)}, connections={len(self.neuronInputList)}, inputCount={len(self.inputNeuronIDs)}, outputCount={len(self.outputNeuronIDs)})"

    # Initialization methods

    def initializeNeurons(self, n: int):
        # Create neurons
        self.neuronList = []
        for i in range(0, n):
            # Initialize random bias
            bias = (random.random()-0.5)*2
            neuron = Neuron(b=bias, activation_fn=ActivationFn.RELU)
            self.neuronList.append(neuron)

    def initializeNeuronConnections(self, n: int):
        apc = generate_non_overlapping_pairs(self.totalNeuronCount)
        if len(apc) < n:
            # error: more connections than possible
            return

        chosenConnections = []
        # Create 'n' random connections between neurons
        i = 0
        while True:
            randId = random.randrange(0, len(apc) - 1 - i)
            conArr = apc.pop(randId)
            if (conArr[0] not in self.outputNeuronIDs) and (conArr[1] not in self.inputNeuronIDs):
                chosenConnections.append(conArr)
                i += 1
            if i >= n:
                break

        for c in chosenConnections:
            nId1 = c[0]-1
            nId2 = c[1]-1
            incomingNeuron = self.neuronList[nId1]
            parentNeuron = self.neuronList[nId2]
            weight = (random.random()-0.5)*2
            neuronInput = NeuronConnection(
                incomingNeuron, parentNeuron, w=weight)
            self.neuronInputList.append(neuronInput)

    # Activation methods

    def computeNeuronValues(self):
        for idx, neuron in enumerate(self.neuronList):
            sum = 0
            if idx in self.inputNeuronIDs:
                inidx = self.inputNeuronIDs.index(idx)
                sum = self.inputNeuronValues[inidx]
            else:
                for neuronInput in self.neuronInputList:
                    if neuronInput.parentNeuron == neuron:
                        sum += neuronInput.value * neuronInput.weight
                sum += neuron.bias
            neuron.value = neuron.activation_fn(sum)

    def computeNeuronInputValues(self):
        for neuronInput in self.neuronInputList:
            neuronInput.value = neuronInput.incomingNeuron.value

    def cycle(self):
        self.computeNeuronValues()
        self.computeNeuronInputValues()

    def setInputNeuronValues(self, array: list[float]):
        self.inputNeuronValues = array

    def getWeights(self):
        out = []
        for el in self.neuronInputList:
            out.append(el.weight)
        return out

    def getBiases(self):
        out = []
        for el in self.neuronList:
            out.append(el.bias)
        return out

    def getNeuronValues(self):
        out = []
        for el in self.neuronList:
            out.append(el.value)
        return out

    def getOutputNeuronValues(self, useSoftmax=True):
        out = []
        for idx in self.outputNeuronIDs:
            out.append(self.neuronList[idx].value)
        if useSoftmax:
            out = softmax(np.array(out))
        return out

    def getConnectionCountPerNeuron(self):
        out = []
        for neuron in self.neuronList:
            count = 0
            for neuronInput in self.neuronInputList:
                if neuronInput.parentNeuron == neuron:
                    count += 1
            out.append(count)
        return out
