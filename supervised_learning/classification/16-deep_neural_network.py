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
        if not isinstance(layers, list) or len(layers) < 1:
            raise TypeError("layers must be an integer")
        if len(layers) < 1:
            raise ValueError("layers must be a positive integer")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            self.weights['W' + str(l)] = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))
