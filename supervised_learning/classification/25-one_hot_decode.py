#!/usr/bin/env python3
"""One Cold Decode?"""
import numpy as np


def one_hot_decode(one_hot):
    """Decode Fun"""
    return np.argmax(one_hot, axis=0)