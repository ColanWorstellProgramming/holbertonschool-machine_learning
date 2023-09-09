#!/usr/bin/env python3
"""Imports"""
import numpy as np


def definiteness(matrix):
    """
    Find Definiteness of matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T):
        return None

    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"
    elif np.all(np.linalg.eigvals(matrix) >= 0):
        return "Positive semi-definite"
    elif np.all(np.linalg.eigvals(matrix) < 0):
        return "Negative definite"
    elif np.all(np.linalg.eigvals(matrix) <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
