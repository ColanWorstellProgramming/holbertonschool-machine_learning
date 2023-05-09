#!/usr/bin/env python3
"""Matric Shape Function"""


def matrix_shape(matrix):
    """Returns the Shape of 2D and 3D Matrices"""
    if len(matrix[0]) == 2:
        return "[{}, {}]".format(len(matrix), len(matrix[0]))
    elif len(matrix[0]) == 3:
        return "[{}, {}, {}]".format(len(matrix), len(matrix[0]), len(matrix[0][0]))
