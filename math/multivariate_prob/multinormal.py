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
            raise ValueError('data must contain multiple data points')

        self.mean = np.mean(data.T, axis=0)
        self.cov = np.matmul((data.T - self.mean.T).T,
                             data.T - self.mean.T) / (self.n - 1)
