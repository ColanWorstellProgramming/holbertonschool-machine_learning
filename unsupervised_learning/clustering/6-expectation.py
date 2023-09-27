#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or not len(X.shape) != 2:
        return None, None

    if  not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    pdf_values = np.zeros((k, n))
    for i in range(k):
        pdf_values[i] = pdf(X, m[i], S[i])

    numerator = pi[:, np.newaxis] * pdf_values

    denominator = np.sum(numerator, axis=0)

    g = numerator / denominator

    log_likelihood = np.log(denominator)
    lll = np.sum(log_likelihood)

    return g, lll
