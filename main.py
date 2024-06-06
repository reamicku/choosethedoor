from neurality import NeuralNet, Distribution, ActivationFn
from neurality.utils import generate_non_overlapping_pairs
from ctdsim.elements import *
import copy
from tqdm import tqdm
import time
import os
import shutil
from datetime import datetime


##### Define values #####
### Simulation variables
n_rooms = 10
n_trapdoors = 0
n_fakedoors = 4
n_creatures = 100

### Simulation variables cont.
n_sims = 10 # Amount of simulations
n_neuralnet_processing_steps = 6 # Neural net gets updated N times before making a decision
n_newnet_creatures_step = 1 # Create new nn every N creatures
n_reproduced = 1 # Select N best networks and use them for reproduction
mutation_rate = 0.001 # Mutation rate every simulation

### Neural Network
n_internal_neurons = 8
n_connections = 42
forbid_direct_io_connections = True
activation_fn = ActivationFn.RELU
random_distrib = Distribution.HE
random_distrib_range = [-4.0, 4.0]

### Misc
save_network_images = True
#########################


timeNow = datetime.now()
timeNowStr = timeNow.strftime("%d-%m-%Y %H:%M:%S")
timeStart = time.perf_counter()

n_alldoors = n_trapdoors + n_fakedoors + 1
n_input_neurons = n_alldoors
n_output_neurons = n_alldoors

generationInfo = []

# Perform N simulations
for j in range(0, n_sims):
    if j != 0: print(f'========== Simulation {j+1}')
    sim = Simulation()

    # Create a simulation with N rooms
    for i in range(0, n_rooms):
        sim.addRoom(n_trapdoors, n_fakedoors)
    
    # Create N creatures
    mut_rate = mutation_rate
    if j == 0: mut_rate = 1.0
    
    for i in tqdm(range(0, n_creatures), desc='Creating creatures'):
        creature = Creature(n_alldoors, n_alldoors)

        if j == 0:
            if i % n_newnet_creatures_step == 0:
                while True:
                    nn = NeuralNet(n_alldoors, n_alldoors, n_internal_neurons, n_connections, activation_fn=activation_fn, forbidDirectIOConnections=forbid_direct_io_connections)
                    if nn.isAllInputOutputConnected():
                        break
            newnn = copy.deepcopy(nn)
        else:
            newnn = copy.deepcopy(bestCreatures[i % n_reproduced]['creature'].nn)
            
        newnn.mutate(mut_rate, distrib=random_distrib, lower=random_distrib_range[0], upper=random_distrib_range[1])
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
    bestCreatures = sim.getBestNCreatures(n_reproduced)
    
    genInfoRow = {
        'room_layout': sim.getRoomsLayoutValues(),
        'creatures_in_rooms': sim.countCreaturesInRooms(),
        'top_creatures': sim.getBestNCreatures(3)
    }
    generationInfo.append(genInfoRow)
    
timeEnd = time.perf_counter()
timeElapsed = round(timeEnd - timeStart, 1)

# Save data
output_dir = f'{timeNow.strftime("%d-%m-%Y_%H-%M-%S")}'

if not os.path.isdir('output'):
    os.mkdir('output')
    
if not os.path.isdir('output/simulations'):
    os.mkdir('output/simulations')
    
os.mkdir(f'output/simulations/{output_dir}')
os.mkdir(f'output/simulations/{output_dir}/images')

print('\nSaving results')

# Prepare markdown template
markdownTemplate = f"""# Simulation

**Started**: <|start_date|>

**Time taken**: <|time_taken|>

## Variables

### Simulation variables

Variable | Value
-|-
Generations | `<|n_sims|>`
Room count | `<|n_rooms|>`
Creature count | `<|n_creatures|>`
Trap doors | `<|n_trapdoors|>`
Fake doors | `<|n_fakedoors|>`
Real doors | `<|n_realdoors|>`
Total doors | `<|n_totaldoors|>`

Variable | Value
-|-
Processing steps per NN | `<|n_neuralnet_processing_steps|>`
New NN interval | `<|n_newnet_creatures_step|>`
Top N reproduced | `<|n_reproduced|>`
Mutation rate | `<|mutation_rate|>`

### Neural Network variables

Variable | Value
-|-
Input neuron count | `<|n_input_neurons|>`
Output neuron count | `<|n_output_neurons|>`
Internal neuron count | `<|n_internal_neurons|>`
Connection count | `<|n_connections|>`
Random distribution range | <|random_distrib_range|>
Random distribution | <|random_distrib|>
Activation Function | <|activation_fn|>
Forbid direct Input-Output connections | <|forbid_direct_io_connections|>

## Results

<|results|>
<|netimages|>"""

markdownReplace = {
    '<|start_date|>': str(timeNowStr),
    '<|time_taken|>': f'{timeElapsed} seconds',
    '<|n_sims|>': str(n_sims),
    '<|n_rooms|>': str(n_rooms),
    '<|n_creatures|>': str(n_creatures),
    '<|n_trapdoors|>': str(n_trapdoors),
    '<|n_fakedoors|>': str(n_fakedoors),
    '<|n_realdoors|>': str(1),
    '<|n_totaldoors|>': str(n_alldoors),
    '<|n_neuralnet_processing_steps|>': str(n_neuralnet_processing_steps),
    '<|n_newnet_creatures_step|>': str(n_newnet_creatures_step),
    '<|n_reproduced|>': str(n_reproduced),
    '<|mutation_rate|>': str(mutation_rate),
    '<|n_input_neurons|>': str(n_input_neurons),
    '<|n_output_neurons|>': str(n_output_neurons),
    '<|n_internal_neurons|>': str(n_internal_neurons),
    '<|n_connections|>': str(n_connections),
    '<|activation_fn|>': activation_fn.__name__,
    '<|random_distrib|>': str(random_distrib),
    '<|random_distrib_range|>': f'`<{random_distrib_range[0]}, {random_distrib_range[1]}>`',
    '<|forbid_direct_io_connections|>': str(forbid_direct_io_connections)
}

# Prepare markdown text
markdownText = markdownTemplate
for key, value in markdownReplace.items():
    markdownText = markdownText.replace(key, value)

markdownTextResults = ''
markdownTextNetImages = '## Neural Network Images'
for i, gen in enumerate(generationInfo):
    textResults = f"""### Generation {i+1}

**Best NNs**: <|best_nns|>

Room | Creatures | Layout
-|-|-<|room_layout|>\n"""

    textNetImages = f"""<|nnets|>"""
    
    textRoomLayout = ''
    for rId in range(n_rooms, 0-1, -1):
        ridStr = rId+1
        if rId == n_rooms:
            ridStr = 'EXIT'
        if rId == n_rooms:
            layoutStr = 'None'
        else:
            layoutStr = gen['room_layout'][rId]
        textRoomLayout += f'\n{ridStr} | `{gen['creatures_in_rooms'][rId]}` | `{layoutStr}`'
    
    textRoomBestNNS = ''
    textNetImagesRow = ''
    for cId, c in enumerate(gen['top_creatures']):
        
        if cId == len(gen['top_creatures'])-1: sep = ''
        else: sep = ', '
        nnName = f'{i+1}_{c['id']}'
        if save_network_images:
            textRoomBestNNS += f'[{nnName}](#{nnName}){sep}'
        else:
            textRoomBestNNS += f'{nnName}{sep}'
        
        if save_network_images:
                if c['creature'].nn.getAllNeuronCount() <= 30:
                    print(f'Saving image of NN-{nnName}')
                    textNetImagesRow += f'\n\n### {nnName}\n\n[Back](#generation-{i+1})\n\n![](./images/{nnName}.png)'
                    c['creature'].nn.saveNetworkImage(f'output/simulations/{output_dir}/images/{nnName}')
    
    textResults = textResults.replace('<|room_layout|>', textRoomLayout)
    textResults = textResults.replace('<|best_nns|>', textRoomBestNNS)
    markdownTextResults += textResults
    textNetImages = textNetImages.replace('<|nnets|>', textNetImagesRow)
    markdownTextNetImages += textNetImages

if not save_network_images: markdownTextNetImages = ''

markdownText = markdownText.replace('<|results|>', markdownTextResults)
markdownText = markdownText.replace('<|netimages|>', markdownTextNetImages)

f = open(f'output/simulations/{output_dir}/index.md', 'w')
f.write(markdownText)
f.close()

print(f'Output at ./output/simulations/{output_dir}/index.md')