#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """

    if not isinstance(X, np.ndarray):
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    if X.ndim != 2:
        return None, None

    if kmin > kmax:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if kmax >= kmin:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

    vars = np.array([variance(X, C) for C, _ in results])
    d_vars = vars[0] - vars

    return results, d_vars
