import os
os.system("cls")

# Forward pass
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b

y = max(z, 0)

dvalue = 1.0

# Optimized backpropegation derivative of sum * derivative of mul * derivative of ReLU in one line per weight and input

drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
drelu_dw0 = dvalue * (1. if z > 0 else 0.) * x[0]
drelu_dx1 = dvalue * (1. if z > 0 else 0.) * w[1]
drelu_dw1 = dvalue * (1. if z > 0 else 0.) * x[1]
drelu_dx2 = dvalue * (1. if z > 0 else 0.) * w[2]
drelu_dw2 = dvalue * (1. if z > 0 else 0.) * x[2]

# Bias

drelu_db = drelu_dz = dvalue * (1. if z > 0 else 0.)

dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db

# Intelligently updating weights and bias

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw0 + xw1 + xw2 + b

y = max(z, 0)
print(y)
