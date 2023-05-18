#!/usr/bin/env python3
"""Poly int"""


def poly_integral(poly, C=0):
    """poly integral, actually pretty fun but hurts jsut a little"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    if not all(isinstance(i, (int, float)) for i in poly):
        return None

    ret = [C]

    for x in range(len(poly)):
        ret.append((poly[x] / (x + 1)))

    return ret
