#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    One HOT Matrix
    """

    return K.utils.to_categorical(labels, classes)
