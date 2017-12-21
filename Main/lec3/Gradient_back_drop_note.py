# @ http://cs231n.github.io/optimization-2/
# set some inputs
import math

x = -2;
y = 5;
z = -4

# perform the forward pass
q = x + y  # q
f = q * z


# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
dfdq = z
dfdz = q

dfdx = 1.0 * dfdq
dfdy = 1.0 * dfdq

# I am just a line-------------------------------------------------------------------


w = [2, -3, -3]
x = [-1, -2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))  # sigmoid function

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f  # gradient ....of the function above
dx = [ddot * w[0], ddot * w[1]]
dw = [ddot * x[0], ddot * x[1]]
