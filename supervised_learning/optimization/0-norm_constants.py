#!/usr/bin/env python3
"""Imports"""
import numpy as np


def normalization_constants(X):
    """Normilization Constants"""
    m = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return m, std
