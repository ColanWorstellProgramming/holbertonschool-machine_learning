#!/usr/bin/env python3
"""Matric Shape Function"""


def matrix_shape(matrix):
    """Returns the Shape of 2D and 3D Matrices"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape