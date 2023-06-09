#!/usr/bin/env python3
"""Poisson"""

e = 2.7182818285
pi = 3.1415926536


class Exponential:
    """Exponential Class"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = lambtha / (sum(data) / len(data))

    def pdf(self, x):
        """pmf calculations"""

        if x < 0:
            return 0

        return (self.lambtha * (e ** (-self.lambtha * x)))

    def cdf(self, x):
        """cdf calculations"""

        if x < 0:
            return 0

        return (1 - (e ** (-self.lambtha * x)))
