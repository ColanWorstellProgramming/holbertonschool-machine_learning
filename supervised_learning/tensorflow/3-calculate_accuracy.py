#!/usr/bin/env python3
"""Calculate Accuracy"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """Calculate Accuracy"""
    for n in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[n], activations[n])

    return x
