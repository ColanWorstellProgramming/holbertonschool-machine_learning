#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    """
    