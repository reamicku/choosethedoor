from neurality import NeuralNet
from ctdsim.elements import *
import copy
from tqdm import tqdm
import time
import os
import shutil

saveNetworkImages = True

sim = Simulation()

# Create a simulation with N rooms
for i in range(0, 10):
    sim.addRoom(0, 4)


# Create N creatures
for i in tqdm(range(0, 500), desc='Creating creatures'):
    creature = Creature(5, 5)

    if i % 50 == 0:
        while True:
            nn = NeuralNet(5, 5, 5, (5+5+5)**1.3, forbidDirectIOConnections=True)
            if nn.isAllInputOutputConnected():
                break

    newnn = copy.deepcopy(nn)
    newnn.mutate(1.0)
    creature.setNeuralNetwork(newnn)
    sim.addCreature(creature)

# Warmup
for i in tqdm(range(0,5), desc='Simulation warmup'):
    sim.step()

# Simulate
for i in tqdm(range(0, 44+1), desc='Simulating'):
    chooseDoor = (i%4==0)
    if i==0:
        chooseDoor = False
    sim.step(chooseDoor=chooseDoor)

sim.printSimulationState()

if os.path.isdir('output/simulation'):
    shutil.rmtree('output/simulation')

if not os.path.isdir('output'):
    os.mkdir('output')
if not os.path.isdir('output/simulation'):
    os.mkdir('output/simulation')
    
f = open('output/simulation/browse.md', 'w')
f.write("# Browse best results")

bestCreatures = sim.getBestCreatureElements()

if len(bestCreatures) < 50:
    for i, el in enumerate(bestCreatures):
        if saveNetworkImages:
            if el['creature'].nn.getAllNeuronCount() <= 64:
                f.write(f'\n\n![](./imgs/bestnetwork{i}.png)')
                el['creature'].nn.saveNetworkImage(f'output/simulation/imgs/bestnetwork{i}')
f.close()
