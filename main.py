from ctdsim.elements import *
import copy
from tqdm import tqdm
import time
import os
from datetime import datetime

from neurality2.neuralnet import NEATPool, NeuralNet, multi_point_crossover, one_point_crossover, uniform_crossover


##### Define values #####
### Simulation variables
n_rooms = 10
n_trapdoors = 0
n_fakedoors = 4
n_creatures = 1000
static_room_layout = True

### Simulation variables cont.
n_generations = 100 # Amount of simulations
n_neuralnet_processing_steps = 4 # Neural net gets updated N times before making a decision
n_newnet_creatures_step = 1 # 1 # Create new nn every N creatures
n_reproduced = n_creatures//50 # Select N best networks and use them for reproduction (dividable by 2!!!)
mutation_rate = 0.01 # Mutation rate every simulation

### Neural Network
n_internal_neurons = 0
n_connection_perc = 0.0

### Misc
save_results = True
save_network_images = False
show_realtime_network_preview = True
hide_empty_rooms = True
#########################


timeNow = datetime.now()
timeNowStr = timeNow.strftime("%d-%m-%Y %H:%M:%S")
timeStart = time.perf_counter()

n_alldoors = n_trapdoors + n_fakedoors + 1
n_input_neurons = n_alldoors
n_output_neurons = n_alldoors

generationInfo = []

sim: Simulation = Simulation()
neatpool = NEATPool()
bestCreatures = []

# Perform N simulations
for j in range(0, n_generations + 1):
    if j != 0:
        print(f'\n========== Generation {j}')
    if j == 0 or (not static_room_layout):
        sim = Simulation()
    elif static_room_layout and isinstance(sim, Simulation):
        sim.creatures = []

    # Create a simulation with N rooms
    if j == 0 or (not static_room_layout):
        for i in range(0, n_rooms):
            sim.addRoom(n_trapdoors, n_fakedoors)

    nn: NeuralNet = NeuralNet(1,1,1,connection_prob=0.0)
    
    if j > 0:
        best_creatures_fitdict = {}
        for i in range(0, len(bestCreatures)):
            fit = bestCreatures[i]['fitness']
            best_creatures_fitdict.update({i: fit*fit})
            
        best_creatures_keys = list(best_creatures_fitdict.keys())
        best_creatures_values = list(best_creatures_fitdict.values())
    
    # print(neatpool)
    
    # Create N creatures
    for i in range(0, n_creatures):
        # First generation
        if j == 0:
            if i % n_newnet_creatures_step == 0:
                nn = NeuralNet(n_alldoors, n_alldoors, n_internal_neurons, connection_prob=n_connection_perc)
            newnn = copy.deepcopy(nn)
            newnn.neat_pool = neatpool
            creature = Creature(n_alldoors, n_alldoors)
            creature.setNeuralNetwork(newnn)
            sim.addCreature(creature)
        # Next generations (children)
        else:
            parent_ids = random.choices(best_creatures_keys, weights=best_creatures_values, k = 2)
            p1id, p2id = parent_ids[0], parent_ids[1]
            p1fit, p2fit = best_creatures_fitdict[p1id], best_creatures_fitdict[p2id]
            p1 = bestCreatures[p1id]['creature'].nn
            p2 = bestCreatures[p2id]['creature'].nn
            
            child_nn = uniform_crossover(p1, p2, p1f=p1fit, p2f=p2fit)
            # child_nn.mutate(mutation_rate)
            
            creature = Creature(n_alldoors, n_alldoors)
            creature.setNeuralNetwork(child_nn)
            sim.addCreature(creature)

    # Simulate
    for i in tqdm(range(0, n_rooms*n_neuralnet_processing_steps+1), desc='Simulating'):
        chooseDoor = (i%n_neuralnet_processing_steps==0)
        if i==0:
            chooseDoor = False
        sim.step(chooseDoor=chooseDoor)

    sim.printSimulationState()
    sim.mutateCreatures(mutation_rate=mutation_rate)
    bestCreatures = sim.getBestNCreatures(n_reproduced)
    
    genInfoRow = {
        'room_layout': sim.getRoomsLayoutValues(),
        'creatures_in_rooms': sim.countCreaturesInRooms(),
        'top_creatures': sim.getBestNCreatures(3),
    }
    print(f'Best fitness: {genInfoRow['top_creatures'][0]['fitness']:.4f}')
    print(f'Best confidence: {100*genInfoRow['top_creatures'][0]['confidence']:.2f}%')
    generationInfo.append(genInfoRow)
    if show_realtime_network_preview:
        genInfoRow['top_creatures'][0]['creature'].nn.save_network_image(f'output/rt')
    
timeEnd = time.perf_counter()
timeElapsed = round(timeEnd - timeStart, 1)

if save_results:

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
    markdownTemplate = """# Simulation

**Started**: <|start_date|>

**Time taken**: <|time_taken|>

## Variables

### Simulation variables

Variable | Value
-|-
Generations | `<|n_generations|>`
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
Activation Function | <|activation_fn|>

## Results

<|results|>
<|netimages|>"""

    markdownReplace = {
        '<|start_date|>': str(timeNowStr),
        '<|time_taken|>': f'{timeElapsed} seconds',
        '<|n_generations|>': str(n_generations),
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
        '<|n_connections|>': str(n_connection_perc),
        '<|activation_fn|>': 'relu',
    }

    # Prepare markdown text
    markdownText = markdownTemplate
    for key, value in markdownReplace.items():
        markdownText = markdownText.replace(key, value)

    markdownTextResults = ''
    markdownTextNetImages = '## Neural Network Images'
    for i, gen in enumerate(generationInfo):
        textResults = f"""### Generation {i}

**Best NNs**: <|best_nns|>

Room | Creatures | Layout
-|-|-<|room_layout|>\n"""

        textNetImages = """<|nnets|>"""
        
        textRoomLayout = ''
        for rId in range(n_rooms, 0-1, -1):
            ridStr = rId+1
            if rId == n_rooms:
                ridStr = 'EXIT'
            if rId == n_rooms:
                layoutStr = 'None'
            else:
                layoutStr = gen['room_layout'][rId]
            cirCount = gen['creatures_in_rooms'][rId]
            if not (cirCount==0 and hide_empty_rooms):
                textRoomLayout += f'\n{ridStr} | `{gen["creatures_in_rooms"][rId]}` | `{layoutStr}`'
        
        textRoomBestNNS = ''
        textNetImagesRow = ''
        for cId, c in enumerate(gen['top_creatures']):
            
            if cId == len(gen['top_creatures'])-1: sep = ''
            else: sep = ', '
            nnName = f'{i}_{c["id"]}'
            if save_network_images:
                textRoomBestNNS += f'[{nnName}](#{nnName}){sep}'
            else:
                textRoomBestNNS += f'{nnName}{sep}'
            
            if save_network_images:
                    if c['creature'].nn.n_total_neurons <= 30:
                        print(f'Saving image of NN-{nnName}')
                        textNetImagesRow += f'\n\n### {nnName}\n\n[Back](#generation-{i})\n\n![](./images/{nnName}.png)'
                        c['creature'].nn.save_network_image(f'output/simulations/{output_dir}/images/{nnName}')
        
        textResults = textResults.replace('<|room_layout|>', textRoomLayout)
        textResults = textResults.replace('<|best_nns|>', textRoomBestNNS)
        markdownTextResults += textResults
        textNetImages = textNetImages.replace('<|nnets|>', textNetImagesRow)
        markdownTextNetImages += textNetImages

    if not save_network_images: markdownTextNetImages = ''

    markdownText = markdownText.replace('<|results|>', markdownTextResults)
    markdownText = markdownText.replace('<|netimages|>', markdownTextNetImages)

    f = open(f'output/simulations/{output_dir}/index.md', 'w', encoding='utf-8')
    f.write(markdownText)
    f.close()

    print(f'Output at ./output/simulations/{output_dir}/index.md')
