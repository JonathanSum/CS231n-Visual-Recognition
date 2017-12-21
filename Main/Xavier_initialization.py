import numpy as np
fan_in, fan_out = 3,4
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)  # layer initialization
W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/2)  # for Relu only