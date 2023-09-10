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

        self.mean = np.mean(data.T, axis=0,)[np.newaxis, :].T
        self.cov = np.matmul((data.T - self.mean.T).T,
                             data.T - self.mean.T) / (self.n - 1)

    def pdf(self, x):
        """
        Calculate PDF at a data point
        """

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d, d2 = x.shape

        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        if d != self.cov.shape[0] or d2 != 1:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        x_minus_mean = x - self.mean
        exponent_term = -0.5 * np.dot(np.dot(x_minus_mean.T,
                                             np.linalg.inv(self.cov)),
                                             x_minus_mean)
        var = np.sqrt(np.linalg.det(self.cov))
        denominator = (2 * np.pi) ** (self.d / 2) * var
        pdf_value = (1 / denominator) * np.exp(exponent_term)

        pdf_value = np.round(pdf_value, 19)
        pdf_value = np.squeeze(pdf_value)

        return pdf_value
