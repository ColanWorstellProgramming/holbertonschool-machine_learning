#!/usr/bin/env python3
"""Sigma Equation"""


def summation_i_squared(n):
    """Addition With Sigma"""
    x = 1
    i = 0

    while x <= n:
        i = i + x ** 2
        x += 1
        print(i)

    return i

