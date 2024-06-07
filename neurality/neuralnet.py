from .neuron import Neuron
from .neuroninput import NeuronConnection
from .math import softmax
from .utils import generate_pairs, generate_non_overlapping_pairs
import random
import numpy as np
from graphviz import Digraph
from .math import *
from enum import Enum
import copy
import math

class ActivationFn(Enum):
    def __str__(self) -> str:
        return self.value.__name__
    
    IDENTITY = identity
    BINARY = binary
    TANH = tanh
    SIGMOID = sigmoid
    RELU = relu
    LEAKY_RELU = leaky_relu


class Distribution(Enum):
    def __str__(self) -> str:
        return self.name
    
    LINEAR = 0
    XAVIER = 1
    HE = 2


class NeuralNet():
    def __init__(self, inputNeuronCount: int, outputNeuronCount: int, neuronInitialCount: int = 64, connectionInitialCount: int = 196, forbidDirectIOConnections: bool = False, activation_fn: ActivationFn = ActivationFn.LEAKY_RELU) -> None:
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
        self.neuronsToInputs = []

        self.activation_fn = activation_fn

        for nIdx in range(0, inputNeuronCount):
            self.inputNeuronIDs.append(nIdx)

        for nIdx in range(self.totalNeuronCount - outputNeuronCount, self.totalNeuronCount):
            self.outputNeuronIDs.append(nIdx)

        self.initializeNeurons(
            inputNeuronCount + outputNeuronCount + neuronInitialCount)
        self.initializeNeuronConnections(
            connectionInitialCount, forbidDirectIOConnections=forbidDirectIOConnections)

        self.setInputNeuronValues(np.zeros(inputNeuronCount))

        self.prepareNeuronInputArray()
        self.mutate(1.0, distrib=Distribution.LINEAR, lower=-1.0, upper=1.0)
        self.weightsBiasRange = [-1.0, 1.0]

    def __str__(self) -> str:
        return f"NeuralNet(size={len(self.neuronList)}, connections={len(self.neuronInputList)}, inputCount={len(self.inputNeuronIDs)}, outputCount={len(self.outputNeuronIDs)})"

    # Initialization methods

    def initializeNeurons(self, n: int):
        # Create neurons
        self.neuronList = []
        for i in range(0, n):
            neuron = Neuron()
            self.neuronList.append(neuron)

    def initializeNeuronConnections(self, n: int, forbidDirectIOConnections: bool = False):
        apc = generate_non_overlapping_pairs(self.totalNeuronCount)
        if len(apc) < n:
            # error: more connections than possible
            return

        chosenConnections = []
        # Create 'n' random connections between neurons
        i = 0
        while True:
            randId = random.randrange(0, len(apc)-1)
            conArr = apc.pop(randId)
            conArrI = [conArr[0]-1, conArr[1]-1]
            if forbidDirectIOConnections:
                if (conArrI[0] not in self.outputNeuronIDs) and (conArrI[1] not in self.inputNeuronIDs) \
                        and not (conArrI[0] in self.inputNeuronIDs and conArrI[1] in self.outputNeuronIDs):
                    chosenConnections.append(conArrI)
                    i += 1
                if i >= n:
                    break
            else:
                if (conArrI[0] not in self.outputNeuronIDs) and (conArrI[1] not in self.inputNeuronIDs):
                    chosenConnections.append(conArrI)
                    i += 1
                if i >= n:
                    break

        for c in chosenConnections:
            nId1 = c[0]
            nId2 = c[1]
            incomingNeuron = self.neuronList[nId1]
            parentNeuron = self.neuronList[nId2]
            neuronInput = NeuronConnection(incomingNeuron, parentNeuron)
            self.neuronInputList.append(neuronInput)

    def prepareNeuronInputArray(self):
        self.neuronsToInputs = []
        for i in range(0, self.getAllNeuronCount()):
            self.neuronsToInputs.append([])
        for i, nc in enumerate(self.neuronInputList):
            nId = self.neuronList.index(nc.parentNeuron)
            self.neuronsToInputs[nId].append(i)

    # Activation methods

    def computeNeuronValues(self, printCalculations: bool = False):
        i = 0
        for el in self.neuronsToInputs:
            sum = 0
            if i in self.inputNeuronIDs:
                inidx = self.inputNeuronIDs.index(i)
                self.neuronList[i].value = self.inputNeuronValues[inidx] + \
                    self.neuronList[i].bias
            else:
                for nInId in el:
                    sum += self.neuronInputList[nInId].value * \
                        self.neuronInputList[nInId].weight
                sum += self.neuronList[i].bias
                self.neuronList[i].value = self.activation_fn(sum)
            i += 1

    def computeNeuronInputValues(self):
        for neuronInput in self.neuronInputList:
            neuronInput.value = neuronInput.incomingNeuron.value

    def cycle(self, printCalculations: bool = False):
        """Performs a computation cycle for the neural net/"""
        self.computeNeuronValues(printCalculations=printCalculations)
        self.computeNeuronInputValues()

    def setInputNeuronValues(self, array: list[float]):
        self.inputNeuronValues = array

    def getWeights(self) -> list[float]:
        """Returns an array of connection weights."""
        out = []
        for el in self.neuronInputList:
            out.append(el.weight)
        return out

    def getBiases(self) -> list[float]:
        """Returns an array of neuron weights."""
        out = []
        for el in self.neuronList:
            out.append(el.bias)
        return out

    def getNeuronValues(self) -> list[float]:
        """Returns an array of output neurons values."""
        out = []
        for el in self.neuronList:
            out.append(el.value)
        return out

    def getOutputNeuronValues(self, useSoftmax=True) -> np.array:
        """Returns an array of softmaxed output neurons values."""
        out = []
        for idx in self.outputNeuronIDs:
            out.append(self.neuronList[idx].value)
        if useSoftmax:
            out = softmax(np.array(out))
        return out

    # Modifying methods

    def mutateWeights(self, mutationRate: float, distrib: Distribution, lower: float, upper: float):
        assert mutationRate >= 0 and mutationRate <= 1, "mutationRate must be a value between 0.0 and 1.0."
        if mutationRate == 0.0:
            return
        
        nCount = self.getAllNeuronConnecionCount()

        if distrib is Distribution.LINEAR:
            randValues = linear_distrib(
                nCount, lower=lower, upper=upper)
        elif distrib is Distribution.XAVIER:
            randValues = xavier_distrib(
                self.getInputNeuronCount(), nCount, lower=lower, upper=upper)
        elif distrib is Distribution.HE:
            randValues = he_distrib(
                nCount, lower=lower, upper=upper)
        
        probs = np.random.rand(nCount)

        for i, neuronInput in enumerate(self.neuronInputList):
            if probs[i] > (1 - mutationRate):
                neuronInput.weight = randValues[i]

    def mutateBiases(self, mutationRate: float, distrib: Distribution, lower: float, upper: float):
        assert mutationRate >= 0 and mutationRate <= 1, "mutationRate must be a value between 0.0 and 1.0."
        if mutationRate == 0.0:
            return
        
        nCount = self.getAllNeuronConnecionCount()

        if distrib is Distribution.LINEAR:
            randValues = linear_distrib(
                nCount, lower=lower, upper=upper)
        elif distrib is Distribution.XAVIER:
            randValues = xavier_distrib(
                self.getInputNeuronCount(), nCount, lower=lower, upper=upper)
        elif distrib is Distribution.HE:
            randValues = he_distrib(
                nCount, lower=lower, upper=upper)
        
        probs = np.random.rand(nCount)

        for i, neuron in enumerate(self.neuronList):
            if probs[i] > (1 - mutationRate):
                neuron.bias = randValues[i]

    def mutate(self, mutationRate: float, distrib: Distribution = Distribution.LINEAR, lower: float = -1.0, upper: float = 1.0):
        assert mutationRate >= 0 and mutationRate <= 1, "mutationRate must be a value between 0.0 and 1.0."
        self.mutateWeights(mutationRate=mutationRate,
                           distrib=distrib, lower=lower, upper=upper)
        self.mutateBiases(mutationRate=mutationRate,
                          distrib=distrib, lower=lower, upper=upper)
        self.weightsBiasRange = [lower, upper]
    
    def onePointCrossover(parent1: 'NeuralNet', parent2: 'NeuralNet', point: float) -> list['NeuralNet']:
        """Performs one-point crossover between self and partner and returns two child networks.
        
        `partner` - Other neural network
        
        `point` - A point where to perform the crossover. Ranges between `<0, 1>`."""
        assert parent1.getAllNeuronCount() == parent2.getAllNeuronCount(), "self and partner must have the same structure"
        assert parent1.getAllNeuronConnecionCount() == parent2.getAllNeuronConnecionCount(), "self and partner must have the same structure"
        
        neuron_count = parent1.getAllNeuronCount()
        nc_count = parent1.getAllNeuronConnecionCount()
        
        neuron_point = math.floor(point * neuron_count)
        nc_point = math.floor(point * nc_count)
        
        child1 = copy.deepcopy(parent2)
        child2 = copy.deepcopy(parent1)
        
        for i in range(neuron_point, neuron_count):
            child1.neuronList[i].bias = parent1.neuronList[i].bias
            child2.neuronList[i].bias = parent2.neuronList[i].bias
        
        for i in range(nc_point, nc_count):
            child1.neuronInputList[i].weight = parent1.neuronInputList[i].weight
            child2.neuronInputList[i].weight = parent2.neuronInputList[i].weight
        
        return [child1, child2]
    
    def kPointCrossover(parent1: 'NeuralNet', parent2: 'NeuralNet', ordered_points: list[float]):
        
        net1 = parent1
        net2 = parent2
        for p in ordered_points:
            net1, net2 = NeuralNet.onePointCrossover(net1, net2, p)
        return net1, net2

    # Checking functions

    def getIncomingNeuronConnectionCount(self, neuronId: int) -> int:
        """Returns neuron of id `neuronId`'s incoming connection count."""

        count = 0
        for neuronInput in self.neuronInputList:
            if neuronInput.parentNeuron == self.neuronList[neuronId]:
                count += 1
        return count

    def getOutcomingNeuronConnectionCount(self, neuronId: int) -> int:
        """Returns neuron of id `neuronId`'s outcoming connection count."""

        count = 0
        for neuronInput in self.neuronInputList:
            if neuronInput.incomingNeuron == self.neuronList[neuronId]:
                count += 1
        return count

    def getAllIncomingNeuronConnectionCount(self) -> list[list[int, int]]:
        """Returns an array of neuron IDs and their incoming connection count.

        Example output: `[[0, 1], [1, 5], [2, 0], ...]`"""

        out = []
        for i, _ in enumerate(self.neuronList):
            count = self.getIncomingNeuronConnectionCount(i)
            out.append([i, count])
        return out

    def getAllOutcomingNeuronConnectionCount(self) -> list[list[int, int]]:
        """Returns an array of neuron IDs and their outgoing connection count.

        Example output: `[[0, 1], [1, 5], [2, 0], ...]`"""

        out = []
        for i, _ in enumerate(self.neuronList):
            count = self.getOutcomingNeuronConnectionCount(i)
            out.append([i, count])
        return out

    def getOrphanNeurons(self):
        """Returns an array of Orphan Neuron IDs.

        Orphan Neurons are neurons that have no incoming or outcoming connections."""
        out = []
        incomingCount = self.getAllIncomingNeuronConnectionCount()
        outcomingCount = self.getAllOutcomingNeuronConnectionCount()
        for i, _ in enumerate(incomingCount):
            if incomingCount[i][1] == 0 and outcomingCount[i][1] == 0:
                out.append(i)
        return out

    def getPurposelessNeurons(self):
        """Returns an array of Purposeless Neuron IDs.

        Purposeless Neurons are neurons that are not output neurons, which have no outcoming connetions."""

        out = []
        outcomingCount = self.getAllOutcomingNeuronConnectionCount()
        for i, _ in enumerate(outcomingCount):
            if outcomingCount[i][1] == 0 and not (i in self.outputNeuronIDs):
                out.append(i)
        return out

    def getInputNeuronConnections(self):
        """Returns an array of Input Neurons with connections."""
        out = []
        for nId in self.inputNeuronIDs:
            neuron = self.neuronList[nId]
            for ni in self.neuronInputList:
                if ni.incomingNeuron == neuron:
                    out.append(nId)
                    break
        return out

    def getOutputNeuronConnections(self):
        """Returns an array of Output Neurons with connections."""
        out = []
        for nId in self.outputNeuronIDs:
            neuron = self.neuronList[nId]
            for ni in self.neuronInputList:
                if ni.parentNeuron == neuron:
                    out.append(nId)
                    break
        return out

    def getInputOutputNeuronConnections(self):
        """Returns an array of Input/Output Neurons with connections"""
        out = []
        inc = self.getInputNeuronConnections()
        onc = self.getOutputNeuronConnections()
        for el in inc:
            out.append(el)
        for el in onc:
            out.append(el)
        return out

    def isAllInputOutputConnected(self):
        return len(self.getInputOutputNeuronConnections()) == (len(self.inputNeuronIDs)+len(self.outputNeuronIDs))

    def getInputNeuronCount(self) -> int: return len(self.inputNeuronIDs)

    def getOutputNeuronCount(self) -> int: return len(self.outputNeuronIDs)

    def getAllNeuronCount(self) -> int: return len(self.neuronList)

    def getAllNeuronConnecionCount(
        self) -> int: return len(self.neuronInputList)

    # Saving the network to an image

    def getNetworkArchitecture(self):
        network = {'layers': [], 'connections': []}
        # insert input neurons
        inputNeuronIDs = []
        for nId in self.inputNeuronIDs:
            inputNeuronIDs.append([f'I{nId}', len(inputNeuronIDs)])
        network['layers'].append({'neurons': inputNeuronIDs})

        # insert output neurons
        outputNeuronIDs = []
        for nId in self.outputNeuronIDs:
            outputNeuronIDs.append([f'O{nId}', len(outputNeuronIDs)])
        network['layers'].append({'neurons': outputNeuronIDs})

        # insert hidden neurons
        hiddenNeuronIDs = []
        for neuron in self.neuronList:
            nId = self.neuronList.index(neuron)
            if not (nId in self.inputNeuronIDs) and not (nId in self.outputNeuronIDs):
                hiddenNeuronIDs.append(
                    [f'H{self.neuronList.index(neuron)}', len(hiddenNeuronIDs)])
        network['layers'].append({'neurons': hiddenNeuronIDs})

        # insert connections
        for nc in self.neuronInputList:
            n1Id = self.neuronList.index(nc.incomingNeuron)
            n2Id = self.neuronList.index(nc.parentNeuron)

            if n1Id in self.inputNeuronIDs:
                n1str = f'I{n1Id}'
            elif n1Id in self.outputNeuronIDs:
                n1str = f'O{n1Id}'
            else:
                n1str = f'H{n1Id}'

            if n2Id in self.inputNeuronIDs:
                n2str = f'I{n2Id}'
            elif n2Id in self.outputNeuronIDs:
                n2str = f'O{n2Id}'
            else:
                n2str = f'H{n2Id}'

            con = (n1str, n2str, nc.weight)
            network['connections'].append(con)

        return network

    def saveNetworkImage(self, filePath: str, format: str = 'png', internalIDs: bool = False):
        """Saves an image of the neural network to a `filepath`."""
        nnNetwork = self.getNetworkArchitecture()

        # Create a new directed graph
        dot = Digraph(comment='Neural Network')

        # Customize node styles
        for i, layer in enumerate(nnNetwork['layers']):
            for neuron in layer['neurons']:
                linewidth = 4
                style = 'filled'
                label = None
                if 'I' in neuron[0]:
                    if not internalIDs:
                        label = f'I{neuron[1]}'
                    dot.node(neuron[0], style=style, shape='circle', color='darkgreen',
                             fillcolor='palegreen', penwidth=f'{linewidth}', label=label)
                elif 'O' in neuron[0]:
                    if not internalIDs:
                        label = f'O{neuron[1]}'
                    dot.node(neuron[0], style=style, shape='circle', color='darkorange',
                             fillcolor='peachpuff', penwidth=f'{linewidth}', label=label)
                else:
                    if not internalIDs:
                        label = ''
                    dot.node(neuron[0], style=style, shape='circle', color='gray40',
                             fillcolor='gray92', penwidth=f'{linewidth}', label=label)

        # Customize edge styles
        for connection in nnNetwork['connections']:
            weight = abs(connection[2]) / (abs(self.weightsBiasRange[1]-self.weightsBiasRange[0])/2)
            linewidth = 0.5 + abs(weight)*3.5
            if connection[2] >= 0:
                dot.edge(connection[0], connection[1],
                         color='blue', penwidth=f'{linewidth}')
            else:
                dot.edge(connection[0], connection[1],
                         color='red', penwidth=f'{linewidth}')

        # Render the graph to a PNG file
        dot.render(filePath, view=False, format=format, cleanup=True)
