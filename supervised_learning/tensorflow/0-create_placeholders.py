#!/usr/bin/env python3
import tensorflow as tf
"""Creating First Tensor Variables"""

def create_placeholders(nx, classes):
    """Create Placeholders"""
    x = tf.cast(tf.Variable(nx, name="x"), dtype=tf.float32)
    y = tf.cast(tf.Variable(classes, name="y"), dtype=tf.float32)
    return x, y
