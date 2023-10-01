#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a
    particular state after a specified number of iterations
    """

    # iterations are more than 0
    if t < 0:
        return None

    # n is the number of states in the markov chain
    n = P.shape[0]

    # Make sure states are the same number
    if P.shape != (n, n) or s.shape != (1, n):
        return None

    current_state = s

    for _ in range(t):
        current_state = np.dot(current_state, P)

    return current_state
