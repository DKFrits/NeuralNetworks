import os
os.system("cls")

import numpy as np

# 1D array = Vector
# Shape = [3, 4]
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# LAYER 1

# 2D+ array = Matrix
# Shape = [3, 4] outside --> inside dim 0 = 3 dim 1 = 4
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# 1D array = Vector
# Shape = [3]
biases = [2, 3, 0.5]

# LAYER 2

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# transpose weights (transpose = switching rows and cols)
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)