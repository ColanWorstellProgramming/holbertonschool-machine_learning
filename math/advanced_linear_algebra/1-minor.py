#!/usr/bin/env python3
"""Imports"""


def minor(matrix):
    """
    Find Minor From Determinant
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for i in range(len(matrix)):
        if not isinstance(matrix[i], list):
            raise TypeError("matrix must be a list of lists")

        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    minor_determ = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j + 1:] for
                          row in (matrix[:i] + matrix[i + 1:])]
            row.append(determinant(sub_matrix))
        minor_determ.append(row)

    return minor_determ


def determinant(matrix):
    """
    Calculate Determinant
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for i in range(len(matrix)):
        if not isinstance(matrix[i], list):
            raise TypeError("matrix must be a list of lists")

        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 0:
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
