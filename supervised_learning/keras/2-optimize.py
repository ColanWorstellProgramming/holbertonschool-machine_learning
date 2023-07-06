#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Optimizer:

    Optimize and compile the network using Adam
    """

    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
