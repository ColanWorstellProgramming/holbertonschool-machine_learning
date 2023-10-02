#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    """
    # n is the number of states in the markov chain
    n = P.shape[0]

    # Make sure states are the same number
    if P.shape != (n, n):
        return None

    if np.any(P <= 0):
        return None

    row_sums = np.sum(P, axis=1)
    if not np.allclose(row_sums, np.ones(n)):
        return None

    eigenvalues, left_eigenvectors = np.linalg.eig(P.T)
    index = np.argmin(np.abs(eigenvalues - 1))
    steady_state = np.real(left_eigenvectors[:, index])
    steady_state /= np.sum(steady_state)

    return steady_state.reshape(1, -1)