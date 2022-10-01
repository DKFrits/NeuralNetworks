import os
os.system("cls")

import numpy as np

# 1D array = Vector
# Shape = [4]
inputs = [1, 2, 3, 2.5]

# 2D+ array = Matrix
# Shape = [3, 4]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# 1D array = Vector
# Shape = [3]
biases = [2, 3, 0.5]

# Matrix before Vector
output = np.dot(weights, inputs) + biases
print(output)