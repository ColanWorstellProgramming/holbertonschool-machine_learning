#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    pi = np.sum(g, axis=1) / n

    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):

        diff = X - m[i, :]
        S[i] = np.dot(g[i, :] * diff.T, diff) / np.sum(g[i, :])

    return pi, m, S
