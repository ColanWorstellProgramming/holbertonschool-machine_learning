#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on dataset

    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA transformation should maintain
    Returns: the weights matrix, W, that maintains var fraction of X‘s original variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality of the transformed X
    """

    # Perform SVD on the covariance matrix
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute the total variance
    total_variance = np.sum(S**2)

    # Calculate the number of dimensions to retain var fraction of the variance
    cumulative_variance = np.cumsum(S**2) / total_variance
    nd = np.argmax(cumulative_variance >= var) + 1

    # Extract the top nd eigenvectors (principal components) from U
    W = Vt[:nd + 1].T

    return W