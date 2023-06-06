#!/usr/bin/env python3
"""One Hot Encode?"""
import numpy as np


def one_hot_encode(Y, classes):
    """Encode Fun"""
    m = Y.shape[0]
    one_hot_matrix = np.zeros((classes, m))

    for i in range(m):
        if Y[i] > 0:
            one_hot_matrix[Y[i], i] = 1

    return one_hot_matrix
