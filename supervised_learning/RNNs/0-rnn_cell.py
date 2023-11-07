#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class RNNCell:
    """
    RNNCell Class
    """
    def __init__(self, i, h, o):
        """
        Constructor
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward Prop
        """
        con = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(con, self.Wh) + self.bh)

        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, y
