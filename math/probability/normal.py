#!/usr/bin/env python3
"""Normal"""

e = 2.7182818285
pi = 3.1415926536


class Normal:
    """Normal Class"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor"""
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.lambtha = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            self.mean = sum(data) / len(data)

            n = 0
            for x in data:
                n += (x - self.mean) ** 2
            self.stddev = ((n / len(data)) ** 0.5)

    def z_score(self, x):
        """z_score"""
        return ((x - self.mean) / self.stddev)

    def x_value(self, z):
        """x_value"""
        return ((self.stddev * z) + self.mean)

    def pdf(self, x):
        """pdf calculations"""

        return (1 / (self.stddev * ((2 * pi) ** 0.5))) * (e ** ( -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))))
