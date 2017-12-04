import numpy as np


class ComputationalGraph(object):
    def forward(self, inputs):
        # 1.(pass inputs to input gates ...)
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss  # the final gate in the graph outputs the loss

    def backward(self):
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward()  # little piece of backprop(chain rule applied)
        return input_gradients
