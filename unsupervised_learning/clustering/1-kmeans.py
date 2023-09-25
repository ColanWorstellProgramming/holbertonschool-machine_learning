#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if type(k) is not int or X.shape[0] <= k or k <= 0:
        return None, None

    centroids = initialize(X, k)

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    cs = np.argmin(distances, axis=1)

    for i in range(iterations):

        cs_cpy = centroids.copy()

        for i in range(len(centroids)):
            if len(X[cs == i] > 0):
                cs_cpy[i] = np.mean(X[cs == i], axis=0)
            else:
                cs_cpy[i] = initialize(X, 1)

        cs = np.argmin(np.linalg.norm(X[:, np.newaxis] - cs_cpy, axis=2),
                       axis=1)

        if np.array_equal(centroids, cs_cpy):
            break

        centroids = cs_cpy

    return centroids, cs


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray):
        return None
    if type(k) is not int or X.shape[0] <= k or k <= 0:
        return None

    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    centroids = np.random.uniform(low=min,
                                  high=max,
                                  size=(k, X.shape[1]))

    return centroids
