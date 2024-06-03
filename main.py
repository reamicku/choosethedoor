import random
from neurality import NeuralNet
from neurality.utils import generate_non_overlapping_pairs
import math

def roundArray(array, digits=3):
    out = []
    for el in array:
        out.append(round(el, digits))
    return out

def randArray(size):
    out = []
    for i in range(0, size):
        out.append((random.random()-0.5)*2)
    return out

# input neurons
inc = 2
# output neurons
onc = 2
# internal neurons
nnc = 6
# neruon connections
nncc = nnc**1.25
# simulation steps
iters = 64
# change the inputs N times over the course of simulating
iarray_nchanges = 8

nn = NeuralNet(inc, onc, nnc, nncc)
if nnc <= 64:
    nn.saveNetworkImage(filePath='output/neural_network', format='png', internalIDs=False)
print(nn)
print(f"Possible pairs for {inc+onc+nnc} neurons: {len(generate_non_overlapping_pairs(inc+onc+nnc))}")
print(f"Orphan neuron count: {len(nn.getOrphanNeurons())}")
print(f"Purposeless neuron count: {len(nn.getPurposelessNeurons())}")

inArray = randArray(inc)
for i in range(0, iters):
    if i%(math.ceil(iters/iarray_nchanges))==0: inArray = randArray(inc)
    nn.setInputNeuronValues(inArray)
    nn.cycle(printCalculations=False)
    if i%(math.ceil(iters/iarray_nchanges))==math.ceil(iters/iarray_nchanges)-1:
        print(f"Cycle: {i+1}\tIN: {roundArray(inArray, 3)}\tOUT: {roundArray(nn.getOutputNeuronValues(), 3)}")