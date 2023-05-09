#!/usr/bin/env python3
def matrix_shape(matrix):
    try:
        return "[{}, {}, {}]".format(len(matrix), len(matrix[0]), len(matrix[0][0]))
    except Exception:
        return "[{}, {}]".format(len(matrix), len(matrix[0]))
