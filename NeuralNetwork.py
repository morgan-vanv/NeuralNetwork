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
        self.structure = [[0]*self.neuronsIL]                                       # Adding Input Layer
        self.structure.extend([[0] * self.neuronsHL] * self.numberHL)               # Adding Hidden Layer/s
        self.structure.append([0] * self.neuronsOL)                                 # Adding Output Layer

        # Initializing Weights and Biases
        i = 1
        self.weights = []
        self.biases = []
        for layer in self.structure:
            while i < self.num_layers:
                self.weights.append([[random.random()] * len(layer)] * len(self.structure[i]))
                self.biases.append([random.random()] * len(self.structure[i]))
                i = i + 1
        print('Initialization Completed')
        
        
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