#!/usr/bin/env python3
"""Imports"""
import numpy as np


def create_momentum_op(loss, alpha, beta1):
    """Create Momentum with Magnitude?"""
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
