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
    f = gzip.open('./trainingData/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

# Reformatting the data into a more usable form
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

# Returns a 10 dimmensional unit vector containing the desired output of the Neural Network
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
