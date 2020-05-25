#
#   @name: Morgan Van Valkenburgh
#   @date: May 20th 2020
#   @brief: 
#

import DataLoader
import NeuralNetwork

training_data, validation_data, test_data = DataLoader.load_data_wrapper()
training_data = list(training_data)

# Notes to remember how to access data
#print(len(training_data)) # Prints how many training entries there are
#print(len(training_data[0][0])) # Prints number of input neurons
#print(len(training_data[0][1])) # Prints number of output neurons
#print(len(training_data[1][1]))

neuronsIL = len(training_data[0][0])    # Setting the number of neurons in the Input Layer from the data
neuronsHL = 16                          # Setting the number of neurons in the Hidden Layer (Arbitrary Choice)
neuronsOL = len(training_data[0][1])    # Setting the number of neurons in the Output Layer from the data
numberHL = 2                            # Setting the number of Hidden Layers (Arbitrary Choice)

net = NeuralNetwork.Network([neuronsIL, neuronsHL, neuronsOL, numberHL])    # Initializing the network
net.print_info()                                                            # Printing information about the network