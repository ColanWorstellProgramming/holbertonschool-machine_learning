#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Building an Projection Block
    """
    F11, F3, F12 = filters

    he_normal = K.initializers.he_normal()

    x = K.layers.Conv2D(F11, (1, 1),
                        strides=s,
                        padding='same',
                        kernel_initializer=he_normal)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F3, (3, 3),
                        padding='same',
                        kernel_initializer=he_normal)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F12,
                        (1, 1),
                        padding='same',
                        kernel_initializer=he_normal)(x)
    x2 = K.layers.BatchNormalization(axis=3)(x)

    x = K.layers.Conv2D(F12,
                        kernel_size=(1, 1),
                        strides=s,
                        padding='same',
                        kernel_initializer=he_normal)(A_prev)
    x3 = K.layers.BatchNormalization(axis=3)(x)

    x = K.layers.Add()([x2, x3])
    x = K.layers.Activation('relu')(x)

    return x
