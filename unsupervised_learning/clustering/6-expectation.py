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
    if not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray):
        return None, None

    if not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray):
        return None, None

    if not len(X.shape) != 2 or len(pi.shape) != 1:
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
