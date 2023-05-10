#!/usr/bin/env python3
"""Matricies 2D Function"""


def add_matrices2D(mat1, mat2):
    """Add 2D Matricies Function"""

    arr3 = [[0 for plc in range(len(mat1))] for plc in range(len(mat1[0]))]

    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat1)):
            for x in range(len(mat1[0])):
                arr3[i][x] = mat1[i][x] + mat2[i][x]
        return arr3
    else:
        return None
