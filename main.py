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
inc = 1
# output neurons
onc = 4
# internal neurons
nnc = 8  
# neruon connections
nncc = 24
# simulation steps
iters = 64
# change the inputs N times over the course of simulating
iarray_nchanges = 8

nn = NeuralNet(inc, onc, nnc, nncc)
nn.saveNetworkImage(filePath='output/neural_network', format='png')
print(nn)
print(f"Possible pairs for {inc+onc+nnc} neurons: {len(generate_non_overlapping_pairs(inc+onc+nnc))}")

inArray = randArray(inc)
for i in range(0, iters):
    if i%(math.ceil(iters/iarray_nchanges))==0: inArray = randArray(inc)
    nn.setInputNeuronValues(inArray)
    nn.cycle()
    if i%(math.ceil(iters/iarray_nchanges))==math.ceil(iters/iarray_nchanges)-1:
        print(f"Cycle: {i+1}\tIN: {roundArray(inArray, 3)}\tOUT: {roundArray(nn.getOutputNeuronValues(), 3)}")