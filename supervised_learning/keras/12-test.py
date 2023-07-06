#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Testing The Model
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
