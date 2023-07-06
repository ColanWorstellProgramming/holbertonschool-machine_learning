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
    save = model

    for i in range(len(layers)):
        if i == 0:
            save = (K.layers.Dense(layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                                  ))(model)
        else:
            save = (K.layers.Dense(layers[i],
                    activation=activations[i],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                                  ))(save)
        if i < len(layers) - 1:
            save = (K.layers.Dropout(1 - keep_prob)(save))

    return K.Model(inputs=model, outputs=save)
