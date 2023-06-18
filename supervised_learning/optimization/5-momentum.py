#!/usr/bin/env python3
"""Imports"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update Variables Momentum"""
    m = beta1 * v + (grad * (1 - beta1))
    v = var - (alpha * m)

    return v, m
