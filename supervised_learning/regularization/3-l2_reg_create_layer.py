#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Using Tensorflow, This function will create a layer
    that uses L2 regularization.
    """
    Given = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    R = tf.contrib.layers.l2_regularizer(scale=lambtha)
    layer = tf.layers.Dense(n, activation, Given, kernel_regularizer=R)

    return layer(prev)
