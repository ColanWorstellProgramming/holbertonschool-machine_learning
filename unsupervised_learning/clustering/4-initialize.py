#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2 or not isinstance(k, int) or k <= 0:
        return None, None, None

    _, d = X.shape

    pi = np.full((k,), 1/k)

    m, _ = kmeans(X, k)

    S = np.zeros((k, d, d))

    for i in range(k):
        S[i] = np.identity(d)

    return pi, m, S
