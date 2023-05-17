#!/usr/bin/env python3
"""Poly div"""


def poly_derivative(poly):
    """poly math, actually pretty fun"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]

    ret = [None for _ in poly[:-1]]
    for x in range(1, len(poly)):
        ret[x-1] = poly[x] * x

    return ret
