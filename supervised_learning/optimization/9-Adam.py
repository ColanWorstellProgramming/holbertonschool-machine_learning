#!/usr/bin/env python3
"""Imports"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update Variables For Adam"""
    return tf.train.RMSPropOptimizer(learning_rate=alpha,
                                     decay=beta2,
                                     epsilon=epsilon).minimize(loss)
