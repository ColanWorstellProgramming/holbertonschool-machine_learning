#!/usr/bin/env python3
"""Imports"""
import numpy as np


def determinant(matrix):
    """
    Calculate Determinant
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    return np.linalg.det(matrix)
