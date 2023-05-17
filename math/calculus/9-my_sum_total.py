#!/usr/bin/env python3
"""Sigma Equation"""


def summation_i_squared(n):
    """Addition With Sigma"""
    if n >= 1:
        return recursive_summation(n, 1, 0)
    else:
        return None


def recursive_summation(n, x, i):
    """helper function"""
    if x <= n:
        i = i + x ** 2
        return recursive_summation(n, x + 1, i)
    else:
        return i
