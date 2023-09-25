#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM
    """
    