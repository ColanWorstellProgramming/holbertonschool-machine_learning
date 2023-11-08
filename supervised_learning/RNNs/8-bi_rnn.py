#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Bi-Directional RNN
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    fw = np.zeros((t + 1, m, h))
    bw = np.zeros((t + 1, m, h))

    fw[0] = h_0
    bw[-1] = h_t

    for step in range(t):
        fw[step + 1] = bi_cell.forward(fw[step], X[step])
        bw[-step - 2] = bi_cell.backward(bw[-step -1], X[-step - 1])

    H = np.concatenate((fw[1:], bw[:-1]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
