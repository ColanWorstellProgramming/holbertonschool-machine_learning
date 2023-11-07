#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class GRUCell:
    """
    GRUCell
    """
    def __init__(self, i, h, o):
        """
        Constructor
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward Prob
        """
        con = np.concatenate((h_prev, x_t), axis=1)

        z = 1 / (1 + np.exp(-(np.dot(con, self.Wz) + self.bz)))

        r = 1 / (1 + np.exp(-(np.dot(con, self.Wr) + self.br)))

        r_h = r * h_prev
        concat_r_h_x = np.concatenate((r_h, x_t), axis=1)
        h_tilde = np.tanh(np.dot(concat_r_h_x, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
