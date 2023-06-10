#!/usr/bin/env python3
import tensorflow as tf
"""Creating First Tensor Variables"""

def create_placeholders(nx, classes):
    """Create Placeholders"""

    x = tf.placeholder("float", None)
    y = tf.placeholder("float", None)

    return x, y
