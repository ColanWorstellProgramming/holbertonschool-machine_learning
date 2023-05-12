#!/usr/bin/env python3
"""Bare Back?"""


def mat_mul(mat1, mat2):
    """Multiply amoung the stars"""

    mat3 = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    if (len(mat1[0]) != len(mat2)):
        return None

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                mat3[i][j] += mat1[i][k] * mat2[k][j]

    return mat3
