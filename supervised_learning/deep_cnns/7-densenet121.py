#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Building a DenseNet-121 Architecture
    """
    he_normal = K.initializers.he_normal()

    inputs = K.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.ReLU()(X)
    X = K.layers.Conv2D(filters=(growth_rate * 2),
                        kernel_size=(7, 7),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=he_normal)(X)
    X = K.layers.MaxPooling2D((3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    X, nb_filters = dense_block(X,
                                (2 * growth_rate),
                                growth_rate,
                                6)

    X, nb_filters = transition_layer(X,
                                     nb_filters,
                                     compression)

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                12)

    X, nb_filters = transition_layer(X,
                                     nb_filters,
                                     compression)

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                24)

    X, nb_filters = transition_layer(X,
                                     nb_filters,
                                     compression)

    X, nb_filters = dense_block(X,
                                nb_filters,
                                growth_rate,
                                16)

    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(1, 1),
                                  padding='valid')(X)

    outputs = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=he_normal)(X)

    return K.models.Model(inputs=inputs, outputs=outputs)
