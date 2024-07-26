"""NeuralNet: A simple implementation of a neural network with customizable architecture.

This module provides a class `NeuralNet` that represents a neural network with a specified
number of input, hidden, and output neurons. The network's connections, weights, and biases
are initialized randomly and can be mutated over time. The module supports forward propagation
to compute the output given an input array, and it provides methods to visualize the network
structure and save it as an image."""

from typing import Union
import itertools
import copy
from graphviz import Digraph
import numpy as np
from .math import relu, softmax


def one_point_crossover(
    parent1: "NeuralNet", parent2: "NeuralNet", point: float
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

    return [child1, child2]


def multi_point_crossover(
    parent1: "NeuralNet", parent2: "NeuralNet", num_points: int
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

    return [child1, child2]


def uniform_crossover(
    p1: "NeuralNet", p2: "NeuralNet", p1f: float = 0.0, p2f: float = 0.0
) -> "NeuralNet":
    assert p1.neat_pool == p2.neat_pool, "Parents are not in the same NEATPool"
    assert p1.neat_pool != None, "Parent 1 is not in a NEATPool"
    assert p2.neat_pool != None, "Parent 2 is not in a NEATPool"
    
    neat_pool = p1.neat_pool
    
    if p1.neat_pool != None and p2.neat_pool != None:
        p1_innovations = p1.neat_pool.get_innovations(p1.connections)
        p1_innovations.sort()
        
        p2_innovations = p2.neat_pool.get_innovations(p2.connections)
        p2_innovations.sort()
    
    if len(p1_innovations) == 0 and len(p2_innovations) == 0:
        if np.random.rand() > 0.5:
            return p1
        else:
            return p2
    
    if len(p1_innovations) == 0:
        max_innovation_id = p2_innovations[-1]
    elif len(p2_innovations) == 0:
        max_innovation_id = p1_innovations[-1]
    else:
        max_innovation_id = max(p1_innovations[-1], p2_innovations[-1])
    
    fitter_parent = p1
    if p1f > p2f:
        fitter_parent_id = 1
    elif p1f < p2f:
        fitter_parent_id = 2
    else:
        fitter_parent_id = np.random.randint(1, 2)
    
    if fitter_parent_id == 1:
        fitter_parent = p1  
    elif fitter_parent_id == 2:
        fitter_parent = p2
    
    n_hidden_new = max(p1.n_hidden, p2.n_hidden)
        
    child = NeuralNet(n_inputs=p1.n_inputs, n_outputs=p1.n_outputs, n_hidden=n_hidden_new, connection_prob=0.0, neat_pool=neat_pool, zeros_init=True)
    
    # print(max_innovation_id, excess_threshold)
    for id in range(0, max_innovation_id):
        is_in_p1i = id in p1_innovations
        is_in_p2i = id in p2_innovations
        
        conn = None
        current_parent = fitter_parent
        
        if is_in_p1i and is_in_p2i:
            conn = neat_pool.get_connection(id)
            if p1f > p2f:
                current_parent = p1
            else:
                current_parent = p2
        else:
            current_parent = fitter_parent
            if fitter_parent_id == 1:
                if is_in_p1i:
                    conn = neat_pool.get_connection(id)
            elif fitter_parent_id == 2:
                if is_in_p2i:
                    conn = neat_pool.get_connection(id)
        
        if conn != None:
            i, j = conn[0], conn[1]
            
            child.connections[i, j] = 1
            child.weights[i, j] = current_parent.weights[i, j]
            child.biases[j] = current_parent.biases[j]
    
    return child


class NEATPool:
    def __init__(self) -> None:
        self.innovations: dict = {}
        self.n_innovations: int = 0
    
    def __str__(self):
        tuples_list = list(self.innovations.values())
        numpy_array = np.array(tuples_list)
        return numpy_array.__str__()
        
    # Check if a connection exists, returns True or False.
    def exists(self, connection: tuple[int, int]) -> bool:
        return connection in self.innovations.values()
    
    # Returns a connection at `i`. Returns False if doesn't exist.
    def get_connection(self, id: int) -> Union[tuple[int, int], None]:
        if id in self.innovations:
            return self.innovations[id]
        else:
            return None
    
    # Returns connection's id. Returns False if doesn't exist.
    def get_id(self, connection: tuple[int, int]) -> Union[int, None]:
        try:
            key = next(key for key, value in self.innovations.items() if value == connection)
        except StopIteration as e:
            return None
        return key
    
    # Puts a new connection into the pool. Returns its `id` or False if already exists.
    def attempt_put(self, connection: tuple[int, int]) -> Union[int, bool]:
        if not self.exists(connection):
            self.innovations.update({self.n_innovations: connection})
            self.n_innovations += 1
            return self.n_innovations - 1
        else:
            return False
        
    def get_innovations(self, connections: np.ndarray) -> list[int]:
        innovation_array: list[int] = []
        for i in range(0, connections.shape[0]):
            for j in range(0, connections.shape[1]):
                if connections[i, j] == 1:
                    innovation_id = self.get_id((i, j))
                    if innovation_id != None:
                        innovation_array.append(innovation_id)                    
        
        return innovation_array


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
        self, n_inputs: int, n_outputs: int, n_hidden: int, connection_prob: float = 0.0, neat_pool: Union[NEATPool, None] = None, zeros_init: bool = False
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
        
        self.n_hidden_max = 10
        
        self.neat_pool = neat_pool

        if not zeros_init:
            self.init_connections(connection_prob)
            self.mutate(1.0, skip_connections=True)

    def init_connections(self, connection_prob: float = 0.0):
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

    def forward(self):
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

    def mutate(
        self,
        mutation_rate: float,
        skip_weights: bool = False,
        skip_biases: bool = False,
        skip_connections: bool = False,
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
        """
        
        shift_chance = 0.80
        shift_max = 0.10
        
        new_neuron_chance = 0.1
        
        # Mutate weights
        if not skip_weights:
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    if np.random.rand() < mutation_rate:
                        if np.random.rand() < shift_chance:
                            self.weights[i, j] += np.random.normal(self.weights.mean(), self.weights.std() * shift_max)
                        else:
                            self.weights[i, j] = np.random.normal(0, 1)
        
        # Mutate biases
        if not skip_biases:
            for i in range(self.biases.shape[0]):
                if np.random.rand() < mutation_rate:
                    if np.random.rand() < shift_chance:
                        self.biases[i] += np.random.normal(self.biases.mean(), self.biases.std() * shift_max)
                    else:
                        self.biases[i] = np.random.normal(0, 1)
        
        # Enable/disable connection
        new_neurons_at_connection_indices = []
        
        if not skip_connections:
            for i in range(self.n_inputs, self.connections.shape[0]):
                for j in itertools.chain(
                    range(0, self.n_inputs),
                    range(self.n_inputs + self.n_outputs, self.n_total_neurons),
                ):
                    if np.random.rand() < mutation_rate:
                        randval = np.random.rand()
                        if randval > new_neuron_chance:
                            self.connections[i, j] = 1 - self.connections[i, j]
                            if self.connections[i, j] == 1 and self.neat_pool != None:
                                self.weights[i, j] = np.random.normal(0, 1)
                                self.neat_pool.attempt_put((i, j))
                        elif randval <= new_neuron_chance \
                                and self.connections[i, j] == 1 \
                                and self.n_hidden + len(new_neurons_at_connection_indices) < self.n_hidden_max:
                            # Add connection indices to a processing list
                            new_neurons_at_connection_indices.append((i, j))
        
        # Create new neurons
        if len(new_neurons_at_connection_indices) > 0:
            for con in new_neurons_at_connection_indices:
                weights_old = self.weights
                biases_old = self.biases
                connections_old = self.connections
                neuron_output_old = self.neuron_output
                
                # resize arrays for adding new neuron
                self.n_hidden += 1
                self.n_total_neurons += 1

                # restore old values
                self.weights: np.ndarray = np.zeros((self.n_total_neurons, self.n_total_neurons))
                self.weights[:weights_old.shape[0], :weights_old.shape[1]] += weights_old
                
                self.biases: np.ndarray = np.zeros(self.n_total_neurons)
                self.biases[:biases_old.shape[0]] += biases_old
                
                self.connections: np.ndarray = np.zeros((self.n_total_neurons, self.n_total_neurons))
                self.connections[:connections_old.shape[0], :connections_old.shape[1]] += connections_old
                
                self.neuron_output: np.ndarray = np.zeros(self.n_total_neurons)
                self.neuron_output[:neuron_output_old.shape[0]] += neuron_output_old
                
                # new connection id
                k = self.n_total_neurons-1
                
                i, j = con[0], con[1]
                
                # Remove i<-j connection and remove weight
                self.connections[i, j] = 0
                self.weights[i, j] = 0
                # Add i<-k and k<-j connection
                self.connections[i, k] = 1
                self.connections[k, j] = 1
                # Set i<-k weight to the previous i<-j weight
                self.weights[i, k] = self.weights[i, j]
                # Set k<-j weight to 1
                self.weights[k, j] = 1.0
                # Set k bias to 0
                self.biases[k] = 0.0
                
                if self.neat_pool != None:
                    self.neat_pool.attempt_put((i, k))
                    self.neat_pool.attempt_put((k, j))

    # Visualizing network

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
            input_neuron_ids.append([f"I{i}", len(input_neuron_ids)])
        network["layers"].append({"neurons": input_neuron_ids})

        # insert output neurons
        output_range = range(self.n_inputs, self.n_inputs + self.n_outputs)
        output_neuron_ids = []
        for i in output_range:
            output_neuron_ids.append([f"O{i}", len(output_neuron_ids)])
        network["layers"].append({"neurons": output_neuron_ids})

        # insert hidden neurons
        hidden_range = range(self.n_inputs + self.n_outputs, self.n_total_neurons)
        hidden_neuron_ids = []
        for i in hidden_range:
            hidden_neuron_ids.append([f"H{i}", len(hidden_neuron_ids)])
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
