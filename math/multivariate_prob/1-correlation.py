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

    if len(C.shape) < 2:
        raise ValueError('C must be a 2D square matrix')

    d, d2 = C.shape

    if d != d2:
        raise ValueError('C must be a 2D square matrix')

    std_dev = np.sqrt(np.diag(C))

    return np.divide(C, np.outer(std_dev, std_dev))
