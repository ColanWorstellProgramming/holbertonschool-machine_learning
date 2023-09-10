#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def correlation(C):
    """
    Calculate Correlation Matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')

    d, d2 = C.shape

    if d != d2:
        raise ValueError('C must be a 2D square matrix')

    mean = np.mean(C, axis=0, keepdims=True)

    std_dev = np.sqrt(np.diag(C))

    # Normally would use np.cov here
    co = np.dot((C - mean).T, C - mean) / (d - 1)

    co /= np.outer(std_dev, std_dev)

    return co
