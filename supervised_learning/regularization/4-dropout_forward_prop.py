#!/usr/bin/env python3
"""Imports"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Drop out regularization method, forward prop function.

    Forward Propagation:

    Z = W * A_prev + b | weighted sum
    A = tanh(Z) | activation function
    A_last = softmax(Z_last) | Softmax used for last layer
    """

    cache = {'A0': X}

    for layer in range(1, L + 1):
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]
        Z = np.matmul(W, cache["A{}".format(layer - 1)]) + b

        if layer == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            dropout_mask = np.random.binomial(n=1, p=keep_prob, size=A.shape)
            A = A * dropout_mask / keep_prob
            cache["D{}".format(layer)] = dropout_mask

        cache["A{}".format(layer)] = A

    return cache
