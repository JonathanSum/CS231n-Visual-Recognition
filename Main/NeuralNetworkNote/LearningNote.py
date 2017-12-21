# assume parameter vector W and its gradient vector dW
import numpy as np
import Note1 as n

x = np.random.randn(3, 1)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(4, 4)
W3 = np.random.randn(4, 4)
b1 = np.random.randn(4, 1)
b2 = np.random.randn(4, 1)
b3 = np.random.randn(4, 1)

a1 = n.Neuron()
a1.weights = W1
a1.bias = b1
print(a1.forward(x))
dW = a1.gradi(x)

param_scale = np.linalg.norm(a1.weights.ravel())
learning_rate = 1e-4
update = -learning_rate * dW  # simple SGD update
update_scale = np.linalg.norm(update.ravel())
a1.weights += update  # the actual update
print(update_scale / param_scale)  # want ~1e-3
