#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Intersection Calculation
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')

    msg = 'x must be an integer that is greater than or equal to 0'

    if not isinstance(x, int) or x < 0:
        raise ValueError(msg)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != len(P.shape):
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError('All values in Pr must be in the range [0, 1]')

    if not np.isclose(sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    factorial = np.math.factorial
    comb = factorial(n) / (factorial(x) * factorial(n - x))
    return Pr * comb * (P ** x) * ((1 - P) ** (n - x))
