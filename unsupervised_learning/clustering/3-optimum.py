#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """
    