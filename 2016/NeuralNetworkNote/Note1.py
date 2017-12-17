import numpy as np
import math


class Neuron(object):
    def forward(self, inputs):
        cell_body_sum = np.sum((np.dot(self.weights,inputs )) + self.bias)
        firing_rate = 1.0 / (1.0 + np.exp(-cell_body_sum))  # sigmoid activation function
        return firing_rate
    def gradi(self, inputs):
        thinking = np.sum(inputs * self.weights) + self.bias
        rate = ((1 - thinking) * (thinking))
        return rate


class Neuron_mine(object):
    def output(self, inputs):
        thinking = np.sum(inputs * self.weights) + self.bias
        making_a_decisions = 1.0 / (1.0 + math.exp(-thinking))
        return making_a_decisions

    def the_rate_of_my_thing(self, inputs):
        thinking = np.sum(inputs * self.weights) + self.bias
        rate = ((1 - thinking) * (thinking))
        return rate


f = lambda x: 1.0 / (1.0 + np.exp(-x))
x = np.random.randn(3, 1)
W1 = np.random.randn(4, 3)
W2 = np.random.randn(4, 4)
W3 = np.random.randn(4, 4)
b1 = np.random.randn(4, 1)
b2 = np.random.randn(4, 1)
b3 = np.random.randn(4, 1)
h1 = f(np.dot(W1, x) + b1)
h2 = f(np.dot(W2, h1) + b2)
out = np.dot(W3, h2) + b3
