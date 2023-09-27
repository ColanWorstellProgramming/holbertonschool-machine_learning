#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)

    lll = 0

    for i in range(iterations + 1):

        g, llll = expectation(X, pi, m, S)

        if abs(llll - lll) <= tol:
            if verbose == True:
                print("Log Likelihood after {} iterations:{}"
                      .format(i, llll.round(5)))

            return pi, m, S, g, llll

        if i < iterations:
            pi, m, S = maximization(X, g)


        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, llll.round(5)))

        lll = llll

    return pi, m, S, g, lll
