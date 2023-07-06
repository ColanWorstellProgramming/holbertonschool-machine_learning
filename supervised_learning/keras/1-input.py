#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    With No Sequential Modle

    Without A Sequential model we can go beyond a plain stack of
    layers where each layer has exactly one input tensor
    and one output tensor.
    """

    model = K.Input(shape=(nx,))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha)
                                 ))

    return K.Model(inputs=model, outputs=x)
