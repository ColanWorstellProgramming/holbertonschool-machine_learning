#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class MultiNormal:
    """
    MultiNormal Class
    """
    def __init__(self, data):
        """
        Constructor
        """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        self.d, self.n = data.shape

        if self.n < 2:
            raise ValueError('data must contain multiple data poitns')

        self.mean = np.mean(data, axis=0, keepdims=True)
        self.cov = np.dot((data - self.mean).T, data - self.mean) / (self.d - 1)
