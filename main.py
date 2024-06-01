import random
from neurality import NeuralNet
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

nn = NeuralNet(4, 4, 512, math.floor(512**1.5))
print(nn)

# print(nn.getConnectionCountPerNeuron())

# weights = nn.getWeights()
# biases = nn.getBiases()
# print(f"{roundArray(weights, 3)}")
# print(f"{roundArray(biases, 3)}")

for i in range(1, 32 + 1):
    inArray = randArray(4)
    nn.setInputNeuronValues(inArray)
    nn.cycle()
    print(f'Cycle = {i} ---------------')
    print(f"IN\t{roundArray(inArray, 3)}")
    print(f"OUT\t{roundArray(nn.getOutputNeuronValues(), 3)}")
    input()