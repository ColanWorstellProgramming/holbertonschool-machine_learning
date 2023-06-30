#!/usr/bin/env python3
"""Imports"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    L2 regularization adds a penalty to the
    loss function that is proportional to the
    square of the model's weights. It discourages
    large weights and encourages small weights without
    enforcing sparsity.

    Now within a Neural Network Using Tensorflow
    """

    return (cost + tf.losses.get_regularization_losses())
