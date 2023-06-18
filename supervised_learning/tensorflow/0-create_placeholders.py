#!/usr/bin/env python3
"""Creating First Tensor Variables"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Create Placeholders"""

    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
