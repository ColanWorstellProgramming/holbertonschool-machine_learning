#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves The Model
    """
    network.save(filename)


def load_model(filename):
    """
    Loads The Model
    """
    return K.models.load_model(filename)
