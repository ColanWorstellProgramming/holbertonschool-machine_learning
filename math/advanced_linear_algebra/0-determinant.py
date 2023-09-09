#!/usr/bin/env python3
"""Imports"""


def determinant(matrix):
    """
    Calculate Determinant
    """

    for i in range(len(matrix)):
        if not isinstance(matrix, list):
            raise TypeError("matrix must be a list of lists")

        if len(matrix) != len(matrix[0]):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determ = 0
    for i in range(len(matrix)):
        sub = [row[:i] + row[i+1:] for row in matrix[1:]]
        cofactor = matrix[0][i] * determinant(sub)
        if i % 2 == 0:
            determ += cofactor
        else:
            determ -= cofactor

    return determ
