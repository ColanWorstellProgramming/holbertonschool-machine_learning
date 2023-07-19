#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Building an Inception Block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    conv_3x3_F3R = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(conv_3x3_F3R)

    conv_5x5_F5R = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(conv_5x5_F5R)

    max_pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    conv_pool = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(max_pool)

    output = K.layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5, conv_pool])

    return output
