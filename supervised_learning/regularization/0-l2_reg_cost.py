#!/usr/bin/env python3
"""Imports"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    L2 regularization adds a penalty to the
    loss function that is proportional to the
    square of the model's weights. It discourages
    large weights and encourages small weights without
    enforcing sparsity.

    L2 Cost = λ * (1/2) * Σ(w²)
    """
    reg_cost = 0

    for l in range(1, L + 1):
        mat = weights["W" + str(l)]
        reg_cost += np.sum(np.square(mat))

    l2_cost = cost + (lambtha / (2 * m)) * reg_cost
    return l2_cost
