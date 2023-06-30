#!/usr/bin/env python3
"""Imports"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Dropout gradient descent is a technique that updates the
    weights of a neural network with L2 regularization. It
    incorporates a regularization term in the weight update
    step, which promotes smaller weights and helps mitigate
    overfitting. By applying dropout, it randomly sets a
    fraction of the neurons to zero during training, which
    enhances the generalization ability of the network.

    W = W - (learning_rate * dW + (learning_rate * lambtha / m) * W)
    """

    m = Y.shape[1]

    for layer in range(L, 0, -1):
        A = cache['A{}'.format(layer)]
        A_prev = cache['A{}'.format(layer - 1)]

        if layer == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
            dZ = (dZ * cache["D{}".format(layer)]) / keep_prob

        W = weights['W{}'.format(layer)]

        dA = np.matmul(W.T, dZ)

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W{}'.format(layer)] -= alpha * dW
        weights['b{}'.format(layer)] -= alpha * db

    return weights
