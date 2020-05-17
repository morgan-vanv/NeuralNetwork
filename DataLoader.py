#
#   @name: Morgan Van Valkenburgh
#   @date: May 14th 2020
#   @brief: This program is tasked with the goal of reading in data from the MNIST database so that our network can use it
#

# Library Imports
import pickle
import gzip
import numpy as np


# Returns data from MNIST dataset as a tuple containing the data that is used for training
def load_data():
    f = gzip.open('../trainingData/train-images.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

