#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class BidirectionalCell:
    """
    BidirectionalCell Class
    """
    def __init__(self, i, h, o):
        """
        Constructor
        """
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Backward
        """
        concat_input = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat_input, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
        Output
        """
        Y = np.matmul(H, self.Wy) + self.by
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)

        return Y
