#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, int):
            raise TypeError("layers must be an integer")
        if layers < 1:
            raise ValueError("layers must be a positive integer")

        self.L = layers
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
