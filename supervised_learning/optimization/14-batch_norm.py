#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Batch Normilization Upgrade"""
    cont = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    den_lay = tf.layers.Dense(inputs=prev, units=n, kernel_initializer=cont)

    z = dense(prev)
    mean, var = tf.nn.moments(z, axes=[0])

    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    Z = tf.nn.batch_normalization(
        z, mean, var, offset=beta, scale=gamma, variance_epsilon=1/100000000
    )

    return activation(Z)
