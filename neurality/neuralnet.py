from .neuron import Neuron, ActivationFn
from .neuroninput import NeuronConnection
from .math import softmax
from .utils import generate_pairs, generate_non_overlapping_pairs
import random
import numpy as np
from graphviz import Digraph


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

        for nIdx in range(0, inputNeuronCount):
            self.inputNeuronIDs.append(nIdx)
        
        for nIdx in range(self.totalNeuronCount - outputNeuronCount - 1, self.totalNeuronCount):
            self.outputNeuronIDs.append(nIdx)

        self.initializeNeurons(
            inputNeuronCount + outputNeuronCount + neuronInitialCount)
        self.initializeNeuronConnections(connectionInitialCount)

        dummyInput = []
        for i in range(0, inputNeuronCount):
            dummyInput.append(0.0)
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
            neuron = Neuron(b=bias, activation_fn=ActivationFn.SIGMOID)
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
            randId = random.randrange(0, len(apc)-1)
            conArr = apc.pop(randId)
            conArrI = [conArr[0]-1, conArr[1]-1]
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
            weight = (random.random()-0.5)*2
            neuronInput = NeuronConnection(
                incomingNeuron, parentNeuron, w=weight)
            self.neuronInputList.append(neuronInput)

    # Activation methods

    def computeNeuronValues(self, printCalculations: bool = False):
        for idx, neuron in enumerate(self.neuronList):
            sum = 0
            sumstr = ''
            if idx in self.inputNeuronIDs:
                inidx = self.inputNeuronIDs.index(idx)
                sum = self.inputNeuronValues[inidx]
                neuron.value = sum
                if printCalculations:
                    print(f'id={idx}; sum = {sum:.3f}')
            else:
                for neuronInput in self.neuronInputList:
                    if neuronInput.parentNeuron == neuron:
                        sum += neuronInput.value * neuronInput.weight
                        if sumstr == '':
                            sumstr = f'{neuronInput.value:.3f}*{neuronInput.weight:.3f}'
                        else:
                            sumstr += f' + {neuronInput.value:.3f}*{neuronInput.weight:.3f}'
                if sum == 0:
                    sumstr = '0'
                sum += neuron.bias
                if printCalculations:
                    print(
                        f'id={idx}; sum = {sumstr} + b = {sum:.3f} -> act(sum) = {neuron.activation_fn(sum):.3f}')
                neuron.value = neuron.activation_fn(sum)

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

    def mutateWeights(self, mutationRate: float):
        assert mutationRate >= 0 and mutationRate <= 1, "mutationRate must be a value between 0.0 and 1.0."
        if mutationRate == 0.0:
            return
        for neuronInput in self.neuronInputList:
            target = (random.random()-0.5)*2
            # linear interpolate between new value and current value
            neuronInput.weight = (1 - mutationRate) * \
                neuronInput.weight + mutationRate*target

    def mutateBiases(self, mutationRate: float):
        assert mutationRate >= 0 and mutationRate <= 1, "mutationRate must be a value between 0.0 and 1.0."
        if mutationRate == 0.0:
            return
        for neuron in self.neuronList:
            target = (random.random()-0.5)*2
            # linear interpolate between new value and current value
            neuron.bias = (1 - mutationRate) * \
                neuron.bias + mutationRate*target

    def mutate(self, mutationRate: float):
        assert mutationRate >= 0 and mutationRate <= 1, "mutationRate must be a value between 0.0 and 1.0."
        self.mutateWeights(mutationRate=mutationRate)
        self.mutateBiases(mutationRate=mutationRate)

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

    def saveNetworkImage(self, filePath: str, format: str, internalIDs: bool = False):
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
            linewidth = 0.5 + abs(connection[2])*3.5
            if connection[2] >= 0:
                dot.edge(connection[0], connection[1],
                         color='blue', penwidth=f'{linewidth}')
            else:
                dot.edge(connection[0], connection[1],
                         color='red', penwidth=f'{linewidth}')

        # Render the graph to a PNG file
        dot.render(filePath, view=False, format=format)
