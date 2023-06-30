#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Using Tensorflow, This function will create a layer
    that uses L2 regularization.
    """
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_regularized_layer = tf.layers.Dense(
        n, activation=activation,
        kernel_regularizer=reg, kernel_initializer=init
    )
    return l2_regularized_layer(prev)
