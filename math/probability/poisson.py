#!/usr/bin/env python3
"""Poisson"""

e = 2.7182818285
pi = 3.1415926536


class Poisson:
    """Poisson Class"""
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """pmf calculations"""

        i = int(k)

        if k < 0:
            return 0

        return ((e ** -self.lambtha) * self.lambtha ** i) / (self.factorial(i))

    def factorial(self, k):
        """factorial helper function"""
        result = 1
        for i in range(1, k+1):
            result *= i
        return result

    def cdf(self, k):
        """cdf calculations"""

        i = int(k)

        if k < 0:
            return 0

        return (self.summation(k) * self.pmf(k))

    def summation(self, k):
        """summation sigma function"""

        j = 0

        for x in range(k + 1):
            j += x

        return j