#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Constructor
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0],
                               bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        """
        mu, sig = self.gp.predict(self.X_s)

        if self.minimize:
            f_best = np.min(self.gp.Y)
            imp = f_best - mu - self.xsi
        else:
            f_best = np.max(self.gp.Y)
            imp = mu - f_best - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sig
            ei = imp * norm.cdf(Z) + sig * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei
