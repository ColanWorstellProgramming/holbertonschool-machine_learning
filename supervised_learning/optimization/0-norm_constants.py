#!/usr/bin/env python3
import numpy as np
"""Imports"""


def normalization_constants(X):
    """Normilization Constants"""
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return m, std