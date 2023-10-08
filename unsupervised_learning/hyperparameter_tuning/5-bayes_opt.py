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

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        """
        X_opt = None
        Y_opt = None

        for _ in range(iterations):
            X_next, _ = self.acquisition()

            if X_next in self.gp.X:
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

            if Y_opt is None or (self.minimize and Y_next < Y_opt):
                X_opt = X_next
                Y_opt = Y_next

            if Y_opt is (not self.minimize and Y_next > Y_opt):
                X_opt = X_next
                Y_opt = Y_next

        self.gp.X = self.gp.X[:-1]

        return X_opt, Y_opt

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
