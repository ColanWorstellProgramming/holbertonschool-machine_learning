#!/usr/bin/env python3
def matrix_transpose(matrix):
    """Transposes 2D matrices"""
    arr = [[0] * len(matrix) for plc in range(len(matrix[0]))]

    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            arr[y][x] = matrix[x][y]

    return arr
