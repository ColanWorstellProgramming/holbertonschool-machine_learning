#!/usr/bin/env python3
"""Calculate Accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate Accuracy"""
    y_max = tf.math.argmax(y, axis=1)
    y_pred_max = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(y_max, y_pred_max)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
