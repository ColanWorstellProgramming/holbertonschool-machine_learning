#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray):
        return None

    if not isinstance(S, np.ndarray) or len(X.shape) != 2:
        return None

    _, d = X.shape

    if m.shape != (d,) or S.shape != (d, d):
        return None

    det_S = np.linalg.det(S)

    if det_S == 0:
        return None

    P = (1 / (np.linalg.det(S) * np.sqrt(2 * np.pi))) ** d

    P = np.maximum(P, 1e-300)

    return P
