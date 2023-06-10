#!/usr/bin/env python3
import tensorflow as tf


def create_placeholders(nx, classes):
    """Create Placeholders"""
    x = tf.Variable(nx, tf.float32)
    y = tf.Variable(classes, tf.float32)
    return x, y
