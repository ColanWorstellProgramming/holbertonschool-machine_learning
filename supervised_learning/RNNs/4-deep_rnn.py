#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Deep RNN Forward Prob
    """
    t, m, _ = X.shape
    lay = len(rnn_cells)
    h = h_0.shape[2]

    H = np.zeros((t + 1, lay, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    H[0] = h_0

    for step in range(t):

        x_t = X[step]

        for layer in range(lay):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]
            h_next, Y[step] = cell.forward(h_prev, x_t)
            H[step + 1, layer] = h_next
            x_t = h_next

    return H, Y
