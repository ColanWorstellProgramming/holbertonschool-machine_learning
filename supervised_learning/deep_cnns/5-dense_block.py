#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Building a Dense Block
    """
    he_normal = K.initializers.he_normal()

    concat_layers = [X]
    nb_filters_concat = nb_filters

    for _ in range(layers):
        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(4 * growth_rate,
                                       (1, 1),
                                       strides=(1, 1),
                                       padding='same',
                                       kernel_initializer=he_normal)(X)

        X = K.layers.BatchNormalization()(X)
        X = K.layers.ReLU()(X)
        X = K.layers.Conv2D(growth_rate,
                                 (3, 3),
                                 strides=(1, 1),
                                 padding='same',
                                 kernel_initializer=he_normal)(X)

        concat_layers.append(X)
        X = K.layers.concatenate(concat_layers,
                                 axis=-1)

        nb_filters_concat += growth_rate

    return X, nb_filters_concat
