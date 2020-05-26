#
#   @name: Morgan Van Valkenburgh
#   @date: May 14th 2020
#   @brief: 
#

# Library Imports
import random
import numpy as np

# File Imports
import DataLoader

class Network(object):

    # Constructor to Initialize the Network
    def __init__(self, network_info):
        # Storing information about Network
        self.neuronsIL = network_info[0]                                            # Stores number of Input Layer Neurons
        self.neuronsHL = network_info[1]                                            # Stores number of Hidden Layer Nuerons
        self.neuronsOL = network_info[2]                                            # Stores number of Output Layer Neurons
        self.numberHL = network_info[3]                                             # Stores number of Hidden Layers
        self.num_layers = 2 + self.numberHL                                         # Stores number of total Layers

        # Assembling Network Structure
        self.structure = []
        self.structure.append([0]*self.neuronsIL)                                   # Adding Input Layer
        if self.numberHL > 1:
            for i in range(self.numberHL):
                self.structure.append([0] * self.neuronsHL)               # Adding Hidden Layer/s
        self.structure.append([0] * self.neuronsOL)                                 # Adding Output Layer

        # Initializing Weights and Biases
        i = 1
        self.weights = []
        self.biases = []

        self.weights.append([[random.random()] * self.neuronsIL] * self.neuronsHL)
        self.biases.append([random.random()] * self.neuronsHL)
        if self.numberHL > 1:
            for i in range(self.numberHL - 1):
                self.weights.append([[random.random()] * self.neuronsHL] * self.neuronsHL)
                self.biases.append([random.random()] * self.neuronsHL)
        self.weights.append([[random.random()] * self.neuronsHL] * self.neuronsOL)
        self.biases.append([random.random()] * self.neuronsOL)

        print('Initialization Completed')

    # Iterates manually through each piece of test data, printing the network output and the expected output  
    def test_function(self, test_data):
        for example in test_data:
            self.render_output(example[0])

    # Calculates the output activation for a single training example
    def render_output(self, test_data):
        # Assign input activations to input layer
        iterator = 0
        for neuron in self.structure[0]:
            self.structure[0][iterator] = float(test_data[0:784][iterator])
            iterator = iterator + 1

        # Loop through, starting at first hidden layer, using weights and biases to calculate activation for each neuron in this layer
        i = 1
        for i in range(self.num_layers):    # Loops through layers (layer iterator)
            if i > 0:                       # Ignores first layer as no calculation is required
                for q in range(len(self.structure[i])): # Loops through neurons in the layer
                    self.structure[i][q] = sigmoid(np.dot(self.weights[i-1][q], self.structure[i-1]) + self.biases[i-1][q])
            
        print(self.structure[len(self.structure)-1])    # Prints output layer

    # Displays information about the arrangement of the network
    def print_info(self):
        print('\nInfo about this network: \n')
        print('# of IL Neurons: ' + str(self.neuronsIL))
        print('# of HL Neurons: ' + str(self.neuronsHL))
        print('# of OL Neurons: ' + str(self.neuronsOL))
        print('# of HLs: ' + str(self.numberHL))

# Math Functions

# Sigmoid Function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Derivative of Sigmoid
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))