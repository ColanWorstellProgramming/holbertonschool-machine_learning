#!/usr/bin/env python3
"""Imports"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle Time"""
    i = np.random.permutation(X.shape[0])
    return X[i], Y[i]
