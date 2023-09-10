#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def mean_cov(X):
    """
    Mean and Covariance
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n, _ = X.shape

    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0)

    co = np.dot((X - mean).T, X - mean) / (n - 1)

    return mean, co
