#!/usr/bin/env python3
"""Matric Shape Function"""


def matrix_shape(matrix):
    """Returns the Shape of 2D and 3D Matrices"""
    try:
        return "[{}, {}, {}]".format(len(matrix), len(matrix[0]), len(matrix[0][0]))
    except Exception:
        return "[{}, {}]".format(len(matrix), len(matrix[0]))
