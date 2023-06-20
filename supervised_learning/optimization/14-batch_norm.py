#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    """Batch Normilization Upgrade"""
    mean = np.mean(Z, axis=0)
    std = np.std(Z, axis=0)
    z = ((Z - mean) / ((std ** 2 + epsilon) ** (1 / 2)))
    return (gamma * z) + beta
