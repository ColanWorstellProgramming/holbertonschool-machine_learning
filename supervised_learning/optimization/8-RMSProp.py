#!/usr/bin/env python3
"""Imports"""
import numpy as np


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Create Variables RMSProp"""
    return tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon).minimize(loss)
