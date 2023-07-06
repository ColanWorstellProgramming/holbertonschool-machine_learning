#!/usr/bin/env python3
"""Imports"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Save The Weights
    """
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load The Weights
    """
    network.load_weights(filepath=filename)