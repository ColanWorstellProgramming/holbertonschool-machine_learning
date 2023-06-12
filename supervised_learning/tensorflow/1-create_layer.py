#!/usr/bin/env python3
"""Creating First Tensor Layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Create Layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init, name=activation)
    return layer
