#!/usr/bin/env python3
"""Imports"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update Variables RMSProp"""
    m = (beta2 * s + ((grad ** 2) * (1 - beta2)))
    v = (var - (alpha * (grad / ((m ** (1/2)) + epsilon))))

    return v, m
