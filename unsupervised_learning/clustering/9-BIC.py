#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
e_m = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a
    GMM using the Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or not isinstance(kmin, int):
        return None, None, None, None

    if kmin <= 0 or not isinstance(iterations, int):
        return None, None, None, None

    if (kmax is not None and (not isinstance(kmax, int) or kmax <= 0)):
        return None, None, None, None

    if kmin >= kmax:
        return None, None, None, None

    if iterations <= 0 or not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n // 2

    best_k = None
    best_result = None
    best_bic = float('inf')
    lll = []
    b = []

    for k in range(kmin, kmax + 1):
        if verbose:
            print("Testing {} clusters".format(k))

        pi, m, S, _, log_likelihood = e_m(X, k, iterations, tol, verbose)

        num_params = k * (d + d + 1) - 1

        bic = num_params * np.log(n) - 2 * log_likelihood

        lll.append(log_likelihood)
        b.append(bic)

        if bic < best_bic:
            best_k = k
            best_result = (pi, m, S)
            best_bic = bic

    return best_k, best_result, np.array(lll), np.array(b)
