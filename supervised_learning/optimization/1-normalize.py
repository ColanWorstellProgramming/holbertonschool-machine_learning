#!/usr/bin/env python3
"""Imports"""
import numpy as np


def normalize(X, m, s):
    """Normilization"""
    return X - m / s
