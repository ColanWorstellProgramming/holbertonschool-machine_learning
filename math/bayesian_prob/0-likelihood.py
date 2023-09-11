#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def likelihood(x, n, P):
    """
    What are the chances?
    """
    if n < 0:
        raise ValueError('n must be a positive integer')

    msg = 'x must be an integer that is greater than or equal to 0'

    if not isinstance(x, int) and x < 0:
        raise ValueError(msg)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray):
        raise TypeError('P must be a 1D numpy.ndarray')

    for y in P:
        if y < 0 or y > 1:
            raise ValueError('All values in P must be in the range [0, 1]')

    return np.array([np.math.comb(n, x) * p**x * (1 - p)**(n - x) for p in P])
