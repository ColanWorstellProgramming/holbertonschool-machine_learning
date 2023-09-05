#!/usr/bin/env python3
"""Imports"""
import numpy as np


def minor(matrix):
    """
    Find Minor
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    size = len(matrix[0])
    matrix = np.array(matrix)
    minor_matrix = np.zeros((size, size), dtype=matrix.dtype)

    for i in range(size):
        for j in range(size):
            sub_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            minor_matrix[i, j] = np.linalg.det(sub_matrix)

    return minor_matrix
