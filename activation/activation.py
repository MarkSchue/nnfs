# First activation
import numpy as np

layer_outputs= np.array([[4.8, 1.21, 2.385],
                         [8.9, -1.81, 0.2],
                         [1.41, 1.051, 0.026],
                         [1.41, 1.051, 0.026]])

print('Sum:', np.sum(layer_outputs))

print('Sum axis 1:', np.sum(layer_outputs, axis=1, keepdims=True))


