import random
from neurality import NeuralNet
from neurality.utils import generate_non_overlapping_pairs
import math
from utils import randArray, roundArray

# input neurons
inc = 5
# output neurons
onc = 5
# internal neurons
nnc = 16
# neruon connections
nncc = nnc**1.3
# simulation steps
iters = 50
# change the inputs N times over the course of simulating
iarray_nchanges = 8

# nn = NeuralNet(inc, onc, nnc, nncc)
while True:
    nn = NeuralNet(inc, onc, nnc, nncc)
    ioconns = nn.getInputOutputNeuronConnections()
    print(ioconns)
    if len(ioconns)==(inc+onc): break

# mdtext = '# Neural Networks\n\n.|1|2|3|4\n-|-|-|-|-'
# f = open('output/test/index.md', "w")
# for i in range(1, 32+1):
#     fpath = f'output/test/imgs/nn{i}'
#     NeuralNet(inc, onc, nnc, nncc).saveNetworkImage(filePath=fpath)
#     if i%4==0:
#         mdtext += f'\n{i//4}|![](./imgs/nn{i-3}.png)|![](./imgs/nn{i-2}.png)|![](./imgs/nn{i-1}.png)|![](./imgs/nn{i-0}.png)'
# f.write(mdtext)

if nnc <= 64:
    nn.saveNetworkImage(filePath='output/neural_network', format='png', internalIDs=False)
print(nn)
print(f"Possible pairs for {inc+onc+nnc} neurons: {len(generate_non_overlapping_pairs(inc+onc+nnc))}")
print(f"Orphan neuron count: {len(nn.getOrphanNeurons())}")
print(f"Purposeless neuron count: {len(nn.getPurposelessNeurons())}")

inArray = randArray(inc)
for j in range(0, 50):
    print(f'Mutation {j}')
    if j>0: nn.mutate(mutationRate=1.0)
    for i in range(0, iters):
        if i%(math.ceil(iters/iarray_nchanges))==0: inArray[0] = random.choice([-1.0, 0.0, 1.0]) #inArray = randArray(inc)
        nn.setInputNeuronValues(inArray)
        nn.cycle(printCalculations=False)
        output = nn.getOutputNeuronValues()
        if i%(math.ceil(iters/iarray_nchanges))==math.ceil(iters/iarray_nchanges)-1:
            print(f"Cycle: {i+1}\tIN: {roundArray(inArray, 3)}\tOUT: {roundArray(nn.getOutputNeuronValues(), 3)}")
