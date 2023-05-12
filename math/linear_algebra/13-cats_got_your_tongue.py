#!/usr/bin/env python3
"""Cats Got Your Tongue"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Got Your Tongue"""
    return np.concatenate((mat1, mat2), axis)
