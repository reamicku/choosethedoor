from neurality import NeuralNet, Distribution, ActivationFn
from neurality.utils import generate_non_overlapping_pairs
from ctdsim.elements import *
import copy
from tqdm import tqdm
import time
import os
import shutil



##### Define values #####
### Simulation variables
n_rooms = 10
n_trapdoors = 0
n_fakedoors = 8
n_creatures = 1000

### Simulation variables cont.
n_neuralnet_processing_steps = 8 # Neural net gets updated N times before making a decision
n_newnet_creatures_step = 10 # Create new nn every N creatures

### Neural Network
n_internal_neurons = 5
n_connections = 80
forbid_direct_io_connections = True
activation_fn = ActivationFn.LEAKY_RELU
random_distrib = Distribution.HE
random_distrib_range = [-4.0, 4.0]
save_network_images = True
#########################


n_alldoors = n_trapdoors + n_fakedoors + 1

sim = Simulation()

# Create a simulation with N rooms
for i in range(0, n_rooms):
    sim.addRoom(n_trapdoors, n_fakedoors)

# Create N creatures
for i in tqdm(range(0, n_creatures), desc='Creating creatures'):
    creature = Creature(n_alldoors, n_alldoors)

    if i % n_newnet_creatures_step == 0:
        while True:
            nn = NeuralNet(n_alldoors, n_alldoors, n_internal_neurons, n_connections, activation_fn=activation_fn, forbidDirectIOConnections=forbid_direct_io_connections)
            if nn.isAllInputOutputConnected():
                break
    newnn = copy.deepcopy(nn)
    newnn.mutate(1.0, distrib=random_distrib, lower=random_distrib_range[0], upper=random_distrib_range[1])
    creature.setNeuralNetwork(newnn)
    sim.addCreature(creature)

# Warmup
for i in tqdm(range(0,n_neuralnet_processing_steps), desc='Simulation warmup'):
    sim.step()

# Simulate
for i in tqdm(range(0, n_rooms*n_neuralnet_processing_steps+1), desc='Simulating'):
    chooseDoor = (i%n_neuralnet_processing_steps==0)
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
        if save_network_images:
            if el['creature'].nn.getAllNeuronCount() <= 30:
                f.write(f'\n\n![](./imgs/bestnetwork{i}.png)')
                el['creature'].nn.saveNetworkImage(f'output/simulation/imgs/bestnetwork{i}')
f.close()
