#!/usr/bin/env python3
"""Creating Forward Prop"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Forward Prop"""
    for n in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[n], activations[n])

    return x
