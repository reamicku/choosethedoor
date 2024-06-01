from .neuron import Neuron
from .neuroninput import NeuronInput
import math
import random
import numpy as np

def generate_pairs(N):
    pairs = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            pairs.append((i, j))
    return pairs

def generate_non_overlapping_pairs(N):
    pairs = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            pairs.append((i, j))
            pairs.append((j, i))
    return pairs

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class NeuralNet():
    def __init__(self, inputNeuronCount: int, outputNeuronCount: int, neuronInitialCount: int = 64, connectionInitialCount: int = 196) -> None:
        assert inputNeuronCount > 0, "inputNeuronCount cannot be less than 1"
        assert outputNeuronCount > 0, "outputNeuronCount cannot be less than 1"
        assert neuronInitialCount > 0, "neuronInitialCount cannot be less than 1"
        self.neuronList = []
        self.neuronInputList = []
        self.inputNeuronIDs = []
        self.outputNeuronIDs = []
        self.totalNeuronCount = inputNeuronCount + outputNeuronCount + neuronInitialCount
        self.inputNeuronValues = []
        
        selectedIDs = []
        i = 0
        while True:
            randId = random.randrange(0, self.totalNeuronCount-1)
            if randId not in selectedIDs:
                self.inputNeuronIDs.append(randId)
                selectedIDs.append(randId)
                i += 1
            if i >= inputNeuronCount: break
        i = 0
        while True:
            randId = random.randrange(0, self.totalNeuronCount-1)
            if randId not in selectedIDs:
                self.outputNeuronIDs.append(randId)
                selectedIDs.append(randId)
                i += 1
            if i >= outputNeuronCount: break
        
        self.initializeNeurons(inputNeuronCount + outputNeuronCount + neuronInitialCount)
        self.initializeNeuronConnections(connectionInitialCount)
        
        dummyInput = []
        for i in range(0, inputNeuronCount): dummyInput.append(0)
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
            neuron = Neuron(b=bias)
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
            if i >= n: break

        for c in chosenConnections:
            nId1 = c[0]-1
            nId2 = c[1]-1
            incomingNeuron = self.neuronList[nId1]
            parentNeuron = self.neuronList[nId2]
            weight = (random.random()-0.5)*2
            neuronInput = NeuronInput(incomingNeuron, parentNeuron, w=weight)
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
            neuron.value = max(0.0, sigmoid(sum))
            # neuron.value = sigmoid(sum)
    
    def computeNeuronInputValues(self):
        for neuronInput in self.neuronInputList:
            neuronInput.value = neuronInput.incomingNeuron.value*1.0
    
    def cycle(self):
        self.computeNeuronValues()
        self.computeNeuronInputValues()
    
    def setInputNeuronValues(self, array: list[float]):
        self.inputNeuronValues = array
    
    def getWeights(self):
        out = []
        for el in self.neuronInputList: out.append(el.weight)
        return out
        
    def getBiases(self):
        out = []
        for el in self.neuronList: out.append(el.bias)
        return out

    def getNeuronValues(self):
        out = []
        for el in self.neuronList: out.append(el.value)
        return out

    def getOutputNeuronValues(self):
        out = []
        for idx in self.outputNeuronIDs: out.append(self.neuronList[idx].value)
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