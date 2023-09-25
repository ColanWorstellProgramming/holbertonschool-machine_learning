#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None

    distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    nearest_centroid_indices = np.argmin(distances, axis=1)

    var = np.sum(distances[np.arange(len(X)), nearest_centroid_indices])

    return var