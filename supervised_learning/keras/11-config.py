#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Save Config
    """
    with open(filename, 'w') as file:
        file.write(network.to_json())
    return None


def load_config(filename):
    """
    Load Config
    """
    with open(filename, 'r') as file:
        return K.models.model_from_json(file.read())
