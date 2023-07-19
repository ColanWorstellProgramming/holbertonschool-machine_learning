#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Building a Dense Block
    """
    he_normal = K.initializers.he_normal()

    for _ in range(layers):
        x = K.layers.BatchNormalization()(X)
        x = K.layers.ReLU()(x)
        x = K.layers.Conv2D(4 * growth_rate,
                            (1, 1),
                            padding='same',
                            kernel_initializer=he_normal)(x)

        x = K.layers.BatchNormalization()(x)
        x = K.layers.ReLU()(x)
        x = K.layers.Conv2D(growth_rate,
                            (3, 3),
                            padding='same',
                            kernel_initializer=he_normal)(x)

        X = K.layers.Concatenate()([X, x])

        nb_filters += growth_rate

    return X, nb_filters
