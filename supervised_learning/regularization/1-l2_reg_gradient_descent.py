#!/usr/bin/env python3
"""Imports"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    L2 gradient descent is a technique that
    updates the weights of a neural network
    with L2 regularization. It adds a regularization
    term to the weight update step, encouraging
    smaller weights and reducing overfitting.

    W -= alpha * (dW + (lambtha / m) * W)
    """

    m = Y.shape[1]

    for layer in range(L, 0, -1):
        A = cache['A{}'.format(layer)]
        A_prev = cache['A{}'.format(layer - 1)]

        if layer == L:
            dZ = A - Y
        else:
            W_next = weights['W{}'.format(layer + 1)]
            dZ = np.dot(W_next.T, dZ) * (1 - np.power(A, 2))

        dW = (1 / m) * np.dot(dZ, A_prev.T) + ((lambtha / m) * weights['W{}'.format(layer)])
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W{}'.format(layer)] -= alpha * dW
        weights['b{}'.format(layer)] -= alpha * db

    return weights