#!/usr/bin/env python3
"""Imports"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Batch Normilization"""
    mean = np.mean(Z, axis=0)
    std = np.std(Z, axis=0)
    z = ((Z - mean) / ((std ** 2 + epsilon) ** (1 / 2)))
    return (gamma * z) + beta
