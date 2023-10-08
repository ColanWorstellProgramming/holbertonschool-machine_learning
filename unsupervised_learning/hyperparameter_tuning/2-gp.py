#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class GaussianProcess:
    """
    GuassianProcess Class
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Represents a noiseless 1D Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process
        """
        self.X = np.append(self.X, X_new)
        self.Y = np.append(self.Y, Y_new)
        self.K = self.kernel(self.X, self.X)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation
        of points in a Gaussian process
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Mean
        mu = np.dot(K_s.T, np.dot(K_inv, self.Y)).flatten()

        # Standard Deviation
        sigma = np.diag(K_ss - np.dot(K_s.T, np.dot(K_inv, K_s)))

        return mu, sigma

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        m, n = X1.shape[0], X2.shape[0]
        K = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                dist = np.linalg.norm(X1[i] - X2[j])
                K[i, j] = self.sigma_f**2 * np.exp(-0.5 * (dist / self.l)**2)

        return K
