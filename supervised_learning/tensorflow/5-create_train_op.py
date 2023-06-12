#!/usr/bin/env python3
"""Train Data"""
import tensorflow as tf


def create_train_op(loss, alpha):
"""Train Data"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
