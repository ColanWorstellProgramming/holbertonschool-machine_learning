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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1 or False in (np.array(layers) > 0):
            raise ValueError("layers must be a positive integer")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if i == 0:
                j = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights['W' + str(i + 1)] = j
            else:
                l = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
                self.weights['W' + str(i + 1)] = l
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
