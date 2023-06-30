#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Using Tensorflow, This function will create a layer
    that uses L2 regularization.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernal_initializer=init,
                            kernel_regularizer=reg)

    return layer(prev)
