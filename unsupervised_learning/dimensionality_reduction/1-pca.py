#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def pca(X, ndim):
    """
    Perform PCA on dataset

    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim)
    containing the transformed version of X
    """
    # Grab Mean
    mean = np.mean(X, axis=0)

    # Perform SVD on the covariance matrix
    _, _, Vt = np.linalg.svd(X - mean)

    # Extract the top ndim eigenvectors (principal components) from U
    W = Vt[:ndim].T

    # Project X onto the reduced dimensionality
    T = np.dot(X - mean, W)

    return T
