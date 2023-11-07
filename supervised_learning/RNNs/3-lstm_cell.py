#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class LSTMCell:
    """
    LSTMCell Class
    """
    def __init__(self, i, h, o):
        """
        Constructor
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward Prob
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        f = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wu) + self.bu)))

        tilde = np.tanh(np.dot(concat_input, self.Wc) + self.bc)
        c_next = f * c_prev + u * tilde

        o = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wo) + self.bo)))

        h_next = o * np.tanh(c_next)

        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, c_next, y
