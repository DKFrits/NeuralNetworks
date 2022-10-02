import os
os.system("cls")

import numpy as np
np.random.seed(0)

# input example data
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# hidden layers

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Don't need transpose because shape is inverted in initialization
        self.output = np.dot(inputs, self.weights) + self.biases

# n_inputs = size of inner array (4 for X)
# n_neurons can be any number
layer1 = Layer_Dense(n_inputs=4, n_neurons=5)

# n_inputs of layer 2 must be the size of neurons of the previous layer, in this case 5
# n_neurons can still be any number
layer2 = Layer_Dense(n_inputs=5, n_neurons=2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)