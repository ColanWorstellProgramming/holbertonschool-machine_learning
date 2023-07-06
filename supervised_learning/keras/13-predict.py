#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Predicting The Model
    """
    return network.predict(x=data, verbose=verbose)
