import os
os.system("cls")

# 1D array = Vector
# Shape = [4]
inputs = [1, 2, 3, 2.5]

# 2D+ array = Matrix
# Shape = [3, 4] outside --> inside
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# 1D array = Vector
# Shape = [3]
biases = [2, 3, 0.5]

# Dot product = Vector * Vector

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for input, weight in zip(inputs, neuron_weights):
        # output = inputs * weights + bias
        neuron_output += input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)