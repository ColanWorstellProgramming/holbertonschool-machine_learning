#!/usr/bin/env python3
"""One Cold Decode?"""
import numpy as np


def one_hot_decode(one_hot):
    """Decode Fun"""

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    labels = np.argmax(one_hot, axis=0)

    return labels
