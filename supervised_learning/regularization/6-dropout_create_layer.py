#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Using Tensorflow, This function will create a layer
    that uses Dropout regularization.
    """
    reg = tf.layers.Dropout(rate=keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(
        inputs=prev,
        units=n,
        activation=activation,
        kernel_regularizer=reg,
        kernel_initializer=init
    )
    return layer
