#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        W = np.random.randn()
        b = 0
        A = 0