import numpy as np
from .math import relu, tanh, softmax
import itertools
from graphviz import Digraph
import copy
import math

class NeuralNet():
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int, connection_prob: float = 0.5) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_total_neurons = self.n_inputs + self.n_outputs + self.n_hidden

        self.weights = np.zeros((self.n_total_neurons, self.n_total_neurons))
        self.biases = np.zeros(self.n_total_neurons)
        self.connections = np.zeros((self.n_total_neurons, self.n_total_neurons))

        self.input_values = np.zeros((self.n_inputs, 1))
        self.output_values = np.zeros((self.n_outputs, 1))
        self.neuron_output = np.zeros(self.n_total_neurons)

        self.init_connections(connection_prob)
        self.mutate(1.0)

    def init_connections(self, connection_prob: float):
        self.connections = np.random.binomial(
            1, 1-connection_prob, (self.n_total_neurons, self.n_total_neurons))
        for i in range(0, self.n_inputs):
            self.connections[i] = np.zeros(self.n_total_neurons)
        
        for i in range(self.n_inputs, self.n_inputs + self.n_outputs):
            for j in range(0, self.n_total_neurons):
                self.connections[j, i] = 0

    def set_input(self, input_array: np.array):
        self.input_values = input_array

    def get_output(self):
        return self.output_values

    def forward(self):
        # Set input values directly without a loop
        self.neuron_output[:self.n_inputs] = self.input_values.flatten()

        # Compute the weighted sum using vectorization
        masked_weights = self.weights * self.connections
        sum_without_biases = np.dot(masked_weights, self.neuron_output)
        
        # Add biases (element-wise operation)
        sum_with_biases = sum_without_biases + self.biases

        # Apply ReLU activation function using NumPy's maximum function
        self.neuron_output = relu(sum_with_biases)

        # Extract the output neurons' activations
        output_activations = self.neuron_output[self.n_inputs:self.n_inputs + self.n_outputs]

        # Apply softmax activation function to the output neurons
        self.output_values = softmax(output_activations).reshape(-1, 1)
    
    def mutate(self, mutation_rate: float, skip_weights: bool = False, skip_biases: bool = False, skip_connections: bool = False, allow_removing_connections: bool = True):
        if not skip_weights:
            for i in range(len(self.weights)):
                if np.random.rand() < mutation_rate:
                    self.weights[i] = np.random.normal(0, 1, self.weights[i].shape)
        if not skip_biases:
            for i in range(len(self.biases)):
                if np.random.rand() < mutation_rate:
                    self.biases[i] = np.random.normal(0, 1, self.biases[i].shape)
        if not skip_connections:
            for i in range(self.n_inputs, self.connections.shape[0]):
                for j in itertools.chain(range(0, self.n_inputs), range(self.n_inputs + self.n_outputs, self.n_total_neurons)):
                    if np.random.rand() < mutation_rate:
                        if allow_removing_connections:
                            self.connections[i, j] = 1 - self.connections[i, j]  # flip connection bit
                        else:
                            self.connections[i, j] = 1
    
    def one_point_crossover(parent1: 'NeuralNet', parent2: 'NeuralNet', point: float) -> list['NeuralNet']:
        assert parent1.n_total_neurons == parent2.n_total_neurons, "self and partner must have the same structure"
        
        neuron_point = int(np.floor(point * parent1.n_total_neurons))
        
        # Create child networks by directly copying the parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Swap biases beyond the crossover point
        child1.biases[neuron_point:] = parent2.biases[neuron_point:]
        child2.biases[neuron_point:] = parent1.biases[neuron_point:]
        
        # Swap weights beyond the crossover point
        # Since the weights are a square matrix, we only need to calculate the point once
        child1.weights[neuron_point:, :] = parent2.weights[neuron_point:, :]
        child2.weights[neuron_point:, :] = parent1.weights[neuron_point:, :]
        
        return [child1, child2]

    def multi_point_crossover(parent1: 'NeuralNet', parent2: 'NeuralNet', num_points: int) -> list['NeuralNet']:
        assert parent1.n_total_neurons == parent2.n_total_neurons, "self and partner must have the same structure"
        
        # Create child networks by directly copying the parents
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Determine crossover points for biases
        crossover_points = np.sort(np.random.choice(parent1.n_total_neurons, num_points + 1, replace=False))
        
        # Perform crossover for biases
        for i in range(len(crossover_points) - 1):
            start, end = crossover_points[i], crossover_points[i + 1]
            child1.biases[start:end] = parent2.biases[start:end]
            child2.biases[start:end] = parent1.biases[start:end]
        
        # Determine crossover points for weights (same points for both dimensions)
        crossover_points = np.sort(np.random.choice(parent1.n_total_neurons, num_points + 1, replace=False))
        
        # Perform crossover for weights
        for i in range(len(crossover_points) - 1):
            start, end = crossover_points[i], crossover_points[i + 1]
            # Swap weights in both dimensions
            child1.weights[start:end, :] = parent2.weights[start:end, :]
            child2.weights[:, start:end] = parent2.weights[:, start:end]
        
        return [child1, child2]
    
    # Visualizing network
    
    def get_network_architecture(self):
        network = {'layers': [], 'connections': []}
        # insert input neurons
        inputRange = range(0, self.n_inputs)
        inputNeuronIDs = []
        for i in inputRange:
            inputNeuronIDs.append([f'I{i}', len(inputNeuronIDs)])
        network['layers'].append({'neurons': inputNeuronIDs})

        # insert output neurons
        outputRange = range(self.n_inputs, self.n_inputs + self.n_outputs)
        outputNeuronIDs = []
        for i in outputRange:
            outputNeuronIDs.append([f'O{i}', len(outputNeuronIDs)])
        network['layers'].append({'neurons': outputNeuronIDs})

        # insert hidden neurons
        hiddenrange = range(self.n_inputs + self.n_outputs, self.n_total_neurons)
        hiddenNeuronIDs = []
        for i in hiddenrange:
            hiddenNeuronIDs.append([f'H{i}', len(hiddenNeuronIDs)])
        network['layers'].append({'neurons': hiddenNeuronIDs})

        # insert connections
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                if self.connections[i, j] == 1:
                    if i in inputRange:
                        n2str = f'I{i}'
                    elif i in outputRange:
                        n2str = f'O{i}'
                    else:
                        n2str = f'H{i}'

                    if j in inputRange:
                        n1str = f'I{j}'
                    elif j in outputRange:
                        n1str = f'O{j}'
                    else:
                        n1str = f'H{j}'
                    
                    con = (n1str, n2str, self.weights[i, j])
                    network['connections'].append(con)
        
        return network
    
    def save_network_image(self, filePath: str, format: str = 'png', internalIDs: bool = False):
        """Saves an image of the neural network to a `filepath`."""
        nnNetwork = self.get_network_architecture()

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
                elif 'H' in neuron[0]:
                    if not internalIDs:
                        label = ''
                    dot.node(neuron[0], style=style, shape='circle', color='gray40',
                             fillcolor='gray92', penwidth=f'{linewidth}', label=label)

        # Customize edge styles
        for connection in nnNetwork['connections']:
            weight = abs(connection[2]) / (abs(5)/2)
            linewidth = 0.5 + abs(weight)*3.5
            if connection[2] >= 0:
                dot.edge(connection[0], connection[1],
                         color='blue', penwidth=f'{linewidth}')
            else:
                dot.edge(connection[0], connection[1],
                         color='red', penwidth=f'{linewidth}')

        # Render the graph to a PNG file
        dot.render(filePath, view=False, format=format, cleanup=True)
