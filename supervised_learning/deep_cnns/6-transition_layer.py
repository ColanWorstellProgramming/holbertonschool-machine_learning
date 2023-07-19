#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Building a Transition Layer
    """
    he_normal = K.initializers.he_normal()

    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.ReLU()(X)

    X = K.layers.Conv2D(nb_filters,
                        (1, 1), strides=(1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(X)

    X = K.layers.AveragePooling2D((2, 2),
                                  strides=(2, 2))(X)

    return X, nb_filters
