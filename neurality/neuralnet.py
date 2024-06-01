from math import floor
import numpy as np
import random
from .neuron import Neuron, InputNeuron, OutputNeuron
from .connection import Connection
from .ntc import NTC

def generate_pairs(N):
    pairs = []
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            pairs.append((i, j))
    return pairs

def sigmoid(x):
    return 1/(1 + np.exp(-x))


class NeuralNet:
    def __init__(self, inputNeurons: int, outputNeurons: int, startingNeuronCount: int = 16, startingConnectionCount: int = 16**1.25) -> None:
        assert startingNeuronCount > 0, "startingNeuronCount cannot be less than 1"
        self.neuronList = []
        self.connectionList = []
        self.ntcList = []
        
        self.startingNeuronCount = startingNeuronCount
        self.inputNeuronList = []
        self.inputNeuronCount = inputNeurons
        self.outputNeuronList = []
        self.outputNeuronCount = outputNeurons
        
        self.initializeNeurons(startingNeuronCount)
        self.initializeConnections(floor(startingConnectionCount))
        
    # IO Methods
    
    def setInput(self, inputNeuronId: int, value: float):
        if inputNeuronId >= self.inputNeuronCount or inputNeuronId < 0:
            # error
            return
        self.inputNeuronList[inputNeuronId] = value
    
    def setInputArray(self, value: list[float]):
        for idx, x in enumerate(value):
            self.input_neuron_list[idx] = x
    
    def getOutput(self, outputNeuronId: int):
        pass # return self.outputNeuronList[outputNeuronId].getValue()
    
    def getOutputArray(self):
        out = []
        for outneuron in self.outputNeuronList:
            pass # out.append(outneuron.getValue())
        return out
    
    # Initialization methods
    
    def __isConnectedToAny(self, nIdx: int) -> Connection:
        n = self.neuronList[nIdx]
        for ntc in self.ntcList:
            if ntc.neuron == n:
                return ntc.connection
        return None
    
    def __areConnected(self, nIdx1: int, nIdx2: int) -> Connection:
        n1 = self.neuronList[nIdx1]
        n2 = self.neuronList[nIdx2]
        for ntc in self.ntcList:
            if ntc.neuron == n1:
                for ntc2 in self.ntcList:
                    if ntc2.neuron == n2 and ntc.connection == ntc2.connection:
                        return ntc.connection
        return None

    def initializeNeurons(self, n: int):
        # Create neurons
        self.neuronList = []
        for i in range(0, n):
            bias = (random.random()-0.5)*2
            neuron = Neuron(b=bias)
            self.neuronList.append(neuron)
        # Create input neurons
        self.inputNeuronList = []
        for i in range(0, self.inputNeuronCount):
            neuron = InputNeuron()
            self.inputNeuronList.append(neuron)
        # Create output neurons
        self.outputNeuronList = []
        for i in range(0, self.outputNeuronCount):
            neuron = OutputNeuron()
            self.outputNeuronList.append(neuron)
    
    def initializeConnections(self, n: int):
        apc = generate_pairs(self.startingNeuronCount)
        if len(apc) < n:
            # error: more connections than possible
            return
        chosenConnections = []
        # Create 'n' random connections between neurons
        for i in range(0, n):
            randIdx = random.randrange(0, len(apc) - 1)
            chosenConnections.append(apc.pop(randIdx))
        
        for c in chosenConnections:
            n1Idx = c[0]-1
            n2Idx = c[1]-1
            n1 = self.neuronList[n1Idx]
            n2 = self.neuronList[n2Idx]
            n1ic = self.__isConnectedToAny(n1Idx)
            n2ic = self.__isConnectedToAny(n2Idx)
            
            if not (n1ic is None) or not (n2ic is None):
                if random.random() > 1-0.08:
                    #   Connect to a random already-existing connection between neurons
                    # instead of creating a new one.
                    if (n1ic is Connection and n2ic is Connection):
                        rand = random.random()
                        if rand < 0.5:
                            con = n1ic
                        else:
                            con = n2ic
                    elif n1ic is Connection:
                        con = n1ic
                    elif n2ic is Connection:
                        con = n2ic
                else:
                    # Create a new connection
                    con = Connection()
            else:
                # Create a new connection
                con = Connection()
            
            bias1 = (random.random()-0.5)*2
            bias2 = (random.random()-0.5)*2
            ntc1 = NTC(n1, con, w=bias1)
            ntc2 = NTC(n2, con, w=bias2)
            
            self.connectionList.append(con)
            self.ntcList.append(ntc1)
            self.ntcList.append(ntc2)
    
    # Activation methods
    
    def computeNTCValues(self):
        for ntc in self.ntcList:
            ntc.value = ntc.connection.value
    
    def computeNeuronValues(self):
        for neuron in self.neuronList:
            sum = 0
            for ntc in self.ntcList:
                sum += ntc.value * ntc.weight
            sum += neuron.bias
            neuron.value = sigmoid(sum)

    def resetConnectionValues(self):
        for con in self.connectionList:
            con.value = 0
    
    def propagateNeuronValuesToConnections(self):
        for ntc in self.ntcList:
            ntc.connection.value += ntc.neuron.value
    
    def cycle(self):
        self.computeNTCValues()
        self.computeNeuronValues()
        self.resetConnectionValues()
        self.propagateNeuronValuesToConnections()