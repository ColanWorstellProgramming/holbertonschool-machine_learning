#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Upgrade Variables For Adam"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
