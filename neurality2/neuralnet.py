"""NeuralNet: A simple implementation of a neural network with customizable architecture.

This module provides a class `NeuralNet` that represents a neural network with a specified
number of input, hidden, and output neurons. The network's connections, weights, and biases
are initialized randomly and can be mutated over time. The module supports forward propagation
to compute the output given an input array, and it provides methods to visualize the network
structure and save it as an image."""

import itertools
import copy
from turtle import clear
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from tqdm import tqdm
from graphviz import Digraph
import numpy as np
from .math import relu, softmax


def one_point_crossover(
    parent1: "NeuralNet", parent2: "NeuralNet", point: float, clear_history: bool = True
) -> list["NeuralNet"]:
    """
    Perform a one-point crossover operation on two neural networks to produce two offspring.

    The crossover point is specified as a float between 0 and 1, which represents the proportion
    of the total number of neurons in the networks. The actual crossover point is determined by
    taking the floor of this value multiplied by the total number of neurons.

    This function assumes that both parent networks have the same structure, meaning they have
    the same number of total neurons. The biases and weights beyond the crossover point are
    swapped between the two parent networks to create two distinct child networks.

    Parameters:
    parent1 (NeuralNet): The first parent neural network.
    parent2 (NeuralNet): The second parent neural network.
    point (float): The crossover point as a proportion of the total number of neurons.

    Returns:
    list of NeuralNet: A list containing the two offspring networks resulting from the crossover.
    """
    assert (
        parent1.n_total_neurons == parent2.n_total_neurons
    ), "self and partner must have the same structure"

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
    
    if clear_history:
        child1.clear_history()
        child2.clear_history()

    return [child1, child2]


def multi_point_crossover(
    parent1: "NeuralNet", parent2: "NeuralNet", num_points: int, clear_history: bool = True
) -> list["NeuralNet"]:
    """
    Perform a multi-point crossover operation on two neural networks to produce two offspring.

    The number of crossover points is specified by the user. At each point, the biases
    and weights of the two parent networks are swapped to create two distinct child networks.
    The crossover points for biases and weights are randomly chosen without replacement,
    ensuring that each segment of the genetic material is swapped only once.

    This function assumes that both parent networks have the same structure, meaning they have
    the same number of total neurons. The biases and weights are swapped at each segment defined
    by the crossover points to create the child networks.

    Parameters:
    parent1 (NeuralNet): The first parent neural network.
    parent2 (NeuralNet): The second parent neural network.
    num_points (int): The number of crossover points to be used for the operation.

    Returns:
    list of NeuralNet: A list containing the two offspring networks resulting from the crossover.
    """
    assert (
        parent1.n_total_neurons == parent2.n_total_neurons
    ), "self and partner must have the same structure"

    # Create child networks by directly copying the parents
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Determine crossover points for biases
    crossover_points = np.sort(
        np.random.choice(parent1.n_total_neurons, num_points + 1, replace=False)
    )

    # Perform crossover for biases
    for i in range(len(crossover_points) - 1):
        start, end = crossover_points[i], crossover_points[i + 1]
        child1.biases[start:end] = parent2.biases[start:end]
        child2.biases[start:end] = parent1.biases[start:end]

    # Determine crossover points for weights (same points for both dimensions)
    crossover_points = np.sort(
        np.random.choice(parent1.n_total_neurons, num_points + 1, replace=False)
    )

    # Perform crossover for weights
    for i in range(len(crossover_points) - 1):
        start, end = crossover_points[i], crossover_points[i + 1]
        # Swap weights in both dimensions
        child1.weights[start:end, :] = parent2.weights[start:end, :]
        child2.weights[start:end, :] = parent1.weights[start:end, :]
        child1.weights[:, start:end] = parent2.weights[:, start:end]
        child2.weights[:, start:end] = parent1.weights[:, start:end]
        
    if clear_history:
        child1.clear_history()
        child2.clear_history()

    return [child1, child2]


class NeuralNet:
    """
    NeuralNet class represents a simple neural network with one hidden layer.

    The network can take in any number of inputs and outputs, but the structure
    of the network (weights and biases) is fixed upon instantiation. The network uses
    ReLU activation for hidden layers and softmax for the output layer.

    Attributes:
        n_inputs (int): The number of input neurons.
        n_outputs (int): The number of output neurons.
        n_hidden (int): The number of hidden neurons.
        n_total_neurons (int): The total number of neurons in the network.
        weights (np.ndarray): A square matrix representing the weights of the connections.
        biases (np.ndarray): An array representing the bias of each neuron.
        connections (np.ndarray): A binary matrix indicating
            the presence of connections between neurons.
        input_values (np.ndarray): The input values to the network.
        output_values (np.ndarray): The output values of the network.
        neuron_output (np.ndarray): The output of the neurons
            after applying the activation function.

    Methods:
        init_connections: Initialize the connections in the network based on a given probability.
        set_input: Set the input values for the network.
        get_output: Retrieve the output values from the network.
        forward: Propagate the input values through the network to compute the output values.
        mutate: Apply a mutation to the weights and biases of the network.
    """

    def __init__(
        self, n_inputs: int, n_outputs: int, n_hidden: int, connection_prob: float = 0.5,
        allow_direct_connections: bool = True
    ) -> None:
        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.n_hidden: int = n_hidden
        self.n_total_neurons: int = self.n_inputs + self.n_outputs + self.n_hidden

        self.weights: np.ndarray = np.zeros((self.n_total_neurons, self.n_total_neurons))
        self.biases: np.ndarray = np.zeros(self.n_total_neurons)
        self.connections: np.ndarray = np.zeros((self.n_total_neurons, self.n_total_neurons))

        self.input_values: np.ndarray = np.zeros((self.n_inputs, 1))
        self.output_values: np.ndarray = np.zeros((self.n_outputs, 1))
        self.neuron_output: np.ndarray = np.zeros(self.n_total_neurons)
        
        self.neuron_output_history: list = []

        self.init_connections(connection_prob, allow_direct_connections=allow_direct_connections)
        self.mutate(1.0, skip_connections=True, allow_direct_connections=allow_direct_connections)

    def init_connections(self, connection_prob: float, allow_direct_connections: bool = False):
        """
        Initialize the connections matrix based on a given connection probability.

        The connections matrix is a binary matrix where a value of 1 indicates
        a connection between neurons, and 0 indicates no connection. The connection
        probablity determines the likelihood of any given connection being established.
        After initializing the connections with random values based on the connection
        probability, we enforce that input neurons do not have any outgoing connections,
        and output neurons do not have incoming connections from hidden or other output
        neurons, nor outgoing connections to any neurons except the output values.
        """
        self.connections: np.ndarray = np.random.binomial(
            1, connection_prob, (self.n_total_neurons, self.n_total_neurons)
        )
        # Enforce no outgoing connections from input neurons
        for i in range(0, self.n_inputs):
            self.connections[i] = np.zeros(self.n_total_neurons)
        # Enforce no incoming connections to output neurons except from hidden neurons
        for i in range(self.n_inputs, self.n_inputs + self.n_outputs):
            for j in range(0, self.n_total_neurons):
                self.connections[j, i] = 0
        if not allow_direct_connections:
            for i in range(self.n_inputs, self.n_inputs + self.n_outputs):
                for j in range(0, self.n_inputs):
                    self.connections[i, j] = 0
        
    def set_input(self, input_array: np.ndarray):
        """
        Set the input values for the network.

        This method takes a one-dimensional numpy array containing the input values.

        Args:
            input_array (np.ndarray): A one-dimensional array of input values
                to be set for the network.
        """
        self.input_values = input_array.reshape(self.n_inputs, -1)

    def get_output(self, apply_softmax: bool = True) -> np.ndarray:
        """
        Retrieve the output values from the network after forward propagation.

        Returns:
            np.ndarray: The output values of the network as a one-dimensional numpy array.
        """
        # Extract the output neurons' activations
        output_activations: np.ndarray = self.neuron_output[
            self.n_inputs : self.n_inputs + self.n_outputs
        ]

        if apply_softmax:
        # Apply softmax activation function to the output neurons
            return softmax(output_activations).reshape(-1, 1)
        else:
            return output_activations

    def forward(self, add_to_history: bool = False):
        """
        Propagate the input values through the network to compute the output values.

        This method performs the forward pass of the neural network. It computes the
        activations of all neurons in the network, starting from the input layer,
        passing through any hidden layers, and ending with the output layer. The activations
        are computed by taking the weighted sum of inputs for each neuron, adding the
        bias, and then applying a non-linear activation function (ReLU for hidden layers
        and softmax for the output layer).
        """
        # Set input values directly without a loop
        self.neuron_output[: self.n_inputs] = self.input_values.flatten()

        sum_with_biases: np.ndarray = np.dot(np.multiply(self.weights, self.connections), self.neuron_output.T) + self.biases

        # Apply ReLU activation function using NumPy's maximum function
        self.neuron_output = relu(sum_with_biases)
                
        if add_to_history:
            neuron_output_temp = self.neuron_output
            neuron_output_temp[: self.n_inputs] = self.input_values.flatten()
            output_activations: np.ndarray = self.neuron_output[
                self.n_inputs : self.n_inputs + self.n_outputs
            ]
            neuron_output_temp[self.n_inputs : self.n_inputs + self.n_outputs] = softmax(output_activations)
            self.neuron_output_history.append(neuron_output_temp)

    def mutate(
        self,
        mutation_rate: float,
        skip_weights: bool = False,
        skip_biases: bool = False,
        skip_connections: bool = False,
        allow_removing_connections: bool = True,
        allow_direct_connections: bool = True
    ):
        """
        Mutate the weights, biases, and connections of the network based on a given mutation rate.

        This method allows for the stochastic modification of the network's parameters.
        Weights and biases can be randomly reinitialized to a normal distribution.
        Connections can be randomly added or removed based on the mutation rate and
        whether removal is allowed. This simulates the process of evolutionary algorithms
        where mutations introduce variability into the population.

        Args:
            mutation_rate (float): The probability with which each parameter
                (weight, bias, connection) will be mutated.
            skip_weights (bool): If True, weights will not be mutated. Default is False.
            skip_biases (bool): If True, biases will not be mutated. Default is False.
            skip_connections (bool): If True, connections will not be mutated. Default is False.
            allow_removing_connections (bool): If True, existing connections can be removed.
                                               If False, connections can only be added.
                                               Default is True.
        """
        if not skip_weights:
            for i, weight_matrix in enumerate(self.weights):
                if np.random.rand() < mutation_rate:
                    self.weights[i] = np.random.normal(0, 1, weight_matrix.shape)
        if not skip_biases:
            for i, bias_vector in enumerate(self.biases):
                if np.random.rand() < mutation_rate:
                    self.biases[i] = np.random.normal(0, 1, bias_vector.shape)
        if not skip_connections:
            for i in range(self.n_inputs, self.connections.shape[0]):
                for j in itertools.chain(
                    range(0, self.n_inputs),
                    range(self.n_inputs + self.n_outputs, self.n_total_neurons),
                ):
                    if np.random.rand() < mutation_rate:
                        if allow_removing_connections:
                            # flip connection bit
                            self.connections[i, j] = 1 - self.connections[i, j]
                        else:
                            self.connections[i, j] = 1
        if not allow_direct_connections:
            for i in range(self.n_inputs, self.n_inputs + self.n_outputs):
                for j in range(0, self.n_inputs):
                    self.connections[i, j] = 0

    # Visualizing network
    
    def clear_history(self):
        self.neuron_output_history = []

    def get_network_architecture(self):
        """Returns an array for further use for `save_network_image` method.

        Returns:
            Array: Neural network architecture data
        """
        network = {"layers": [], "connections": []}
        # insert input neurons
        input_range = range(0, self.n_inputs)
        input_neuron_ids = []
        for i in input_range:
            input_neuron_ids.append([f"I{i}", len(input_neuron_ids), i])
        network["layers"].append({"neurons": input_neuron_ids})

        # insert output neurons
        output_range = range(self.n_inputs, self.n_inputs + self.n_outputs)
        output_neuron_ids = []
        for i in output_range:
            output_neuron_ids.append([f"O{i}", len(output_neuron_ids), i])
        network["layers"].append({"neurons": output_neuron_ids})

        # insert hidden neurons
        hidden_range = range(self.n_inputs + self.n_outputs, self.n_total_neurons)
        hidden_neuron_ids = []
        for i in hidden_range:
            hidden_neuron_ids.append([f"H{i}", len(hidden_neuron_ids), i])
        network["layers"].append({"neurons": hidden_neuron_ids})

        # insert connections
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                if self.connections[i, j] == 1:
                    if i in input_range:
                        n2str = f"I{i}"
                    elif i in output_range:
                        n2str = f"O{i}"
                    else:
                        n2str = f"H{i}"

                    if j in input_range:
                        n1str = f"I{j}"
                    elif j in output_range:
                        n1str = f"O{j}"
                    else:
                        n1str = f"H{j}"

                    con = (n1str, n2str, self.weights[i, j])
                    network["connections"].append(con)

        return network

    def save_network_image(
        self,
        file_path: str,
        image_format: str = "png",
        display_internal_ids: bool = False,
    ):
        """Saves an image of the neural network to a `filepath`."""
        nn_network_arch = self.get_network_architecture()

        # Create a new directed graph
        dot = Digraph(comment="Neural Network")

        # Customize node styles
        for layer in nn_network_arch["layers"]:
            for neuron in layer["neurons"]:
                linewidth = 4
                style = "filled"
                label = None
                if "I" in neuron[0]:
                    if not display_internal_ids:
                        label = f"I{neuron[1]}"
                    dot.node(
                        neuron[0],
                        style=style,
                        shape="circle",
                        color="darkgreen",
                        fillcolor="palegreen",
                        penwidth=f"{linewidth}",
                        label=label,
                    )
                elif "O" in neuron[0]:
                    if not display_internal_ids:
                        label = f"O{neuron[1]}"
                    dot.node(
                        neuron[0],
                        style=style,
                        shape="circle",
                        color="darkorange",
                        fillcolor="peachpuff",
                        penwidth=f"{linewidth}",
                        label=label,
                    )
                elif "H" in neuron[0]:
                    if not display_internal_ids:
                        label = ""
                    dot.node(
                        neuron[0],
                        style=style,
                        shape="circle",
                        color="gray40",
                        fillcolor="gray92",
                        penwidth=f"{linewidth}",
                        label=label,
                    )

        # Customize edge styles
        for connection in nn_network_arch["connections"]:
            weight = abs(connection[2]) / (abs(5) / 2)
            linewidth = 0.5 + abs(weight) * 3.5
            if connection[2] >= 0:
                dot.edge(
                    connection[0], connection[1], color="blue", penwidth=f"{linewidth}"
                )
            else:
                dot.edge(
                    connection[0], connection[1], color="red", penwidth=f"{linewidth}"
                )

        # Render the graph to a PNG file
        dot.render(file_path, view=False, format=image_format, cleanup=True)
        
    
    def save_network_neuron_activation(self, activations: np.ndarray, file_path: str, display_internal_ids: bool = False, image_format: str = "png"):
        nn_network_arch = self.get_network_architecture()
        # Create a new directed graph
        dot = Digraph(comment="Neural Network", strict=True)
        
        weight_max = 3.5 #np.max(activations)

        # Customize node styles
        for layer in nn_network_arch["layers"]:
            for neuron in layer["neurons"]:
                linewidth = 4
                style = "filled"
                label = None
                
                weight = activations[neuron[2]]
                weightn = abs(weight/weight_max)
                fillcolor = f"{240.0/360:.3f} {weightn:.3f} 1.000" if activations[neuron[2]] > 0 else f"0.000 {weightn:.3f} 1.000"
                if "I" in neuron[0]:
                    weightn = abs(weight/1.0)
                    fillcolor = f"{240.0/360:.3f} {weightn:.3f} 1.000" if activations[neuron[2]] > 0 else f"0.000 {weightn:.3f} 1.000"
                    if not display_internal_ids:
                        label = f"I{neuron[1]}"
                    dot.node(
                        neuron[0],
                        style=style,
                        shape="circle",
                        color="darkgreen",
                        fillcolor=fillcolor,
                        penwidth=f"{linewidth}",
                        label=label,
                    )
                elif "O" in neuron[0]:
                    weightn = abs(weight/1.0)
                    fillcolor = f"{240.0/360:.3f} {weightn:.3f} 1.000" if activations[neuron[2]] > 0 else f"0.000 {weightn:.3f} 1.000"
                    if not display_internal_ids:
                        label = f"O{neuron[1]}"
                    dot.node(
                        neuron[0],
                        style=style,
                        shape="circle",
                        color="darkorange",
                        fillcolor=fillcolor,
                        penwidth=f"{linewidth}",
                        label=label,
                    )
                elif "H" in neuron[0]:
                    if not display_internal_ids:
                        label = ""
                    dot.node(
                        neuron[0],
                        style=style,
                        shape="circle",
                        color="gray40",
                        fillcolor=fillcolor,
                        penwidth=f"{linewidth}",
                        label=label,
                    )

        # Customize edge styles
        for connection in nn_network_arch["connections"]:
            dot.edge(
                connection[0], connection[1], color="gray50", penwidth=f"{1}"
            )

        # Render the graph to a PNG file
        dot.render(file_path, view=False, format=image_format, cleanup=True)
    
    
    def save_network_neuron_activation_images(self, output_dir: str):
        for i in tqdm(range(len(self.neuron_output_history)), desc='Saving activations'):
            actv = self.neuron_output_history[i]
            self.save_network_neuron_activation(actv, f'{output_dir}/act_{i}')


    def save_network_image_networkx(self, output_path):
        """Visualize the neural network using NetworkX."""
        G = nx.DiGraph()

        # Add input neurons
        G.add_nodes_from(range(self.n_inputs), neuron_type='input', subset=2)

        # Add output neurons
        G.add_nodes_from(range(self.n_inputs, self.n_inputs + self.n_outputs), neuron_type='output', subset=0)
        
        # Add hidden neurons
        G.add_nodes_from(range(self.n_inputs + self.n_outputs, self.n_total_neurons), neuron_type='hidden', subset=1)

        # Add connections as edges
        for i in range(self.connections.shape[0]):
            for j in range(self.connections.shape[1]):
                if self.connections[i, j] == 1:
                    G.add_edge(j, i, weight=-self.weights[i, j])

        # Draw the graph
        subset_key = {
            0: [n for n in G.nodes if G.nodes[n]['subset']==0],
            1: [n for n in G.nodes if G.nodes[n]['subset']==1],
            2: [n for n in G.nodes if G.nodes[n]['subset']==2]
        }
        # pos = nx.multipartite_layout(G, subset_key=subset_key, scale=-1)
        # pos = nx.spring_layout(G, k=6, seed=0)
        pos = graphviz_layout(G, prog='dot')
        node_colors = ['lightblue' if G.nodes[n]['neuron_type'] == 'input' else
                        'orange' if G.nodes[n]['neuron_type'] == 'output' else
                        'lightgrey' for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_labels(G, pos)
        
        max_weight = np.amax(self.weights)
        min_weight = np.amin(self.weights)
        norm = mpl.colors.Normalize(min_weight, max_weight)
        colormap = mpl.colormaps['RdBu']

        # Create a list of colors for each edge based on weight
        edge_colors = [colormap(norm(G.edges[u, v]['weight'])) for u, v in G.edges]
        nx.draw_networkx_edges(G, pos, connectionstyle='arc3', width=0.85, node_size=300, style='solid', arrowsize=8, arrows=True, arrowstyle='-|>', edge_color=edge_colors)

        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()