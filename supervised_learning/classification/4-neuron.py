#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class Neuron:
    """Neuron Class"""
    def __init__(self, nx):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """Forward Propogation"""
        self.__A = self.sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A

    def sigmoid(self, X):
        """Sigmoid Helper"""
        return 1 / (1 + np.exp(-X))

    def cost(self, Y, A):
        """Logistical Regression Cost Func"""
        m = Y.shape[1]
        j = np.log(1.0000001 - A)
        return ((-1/m) * np.sum(Y * np.log(A) + (1 - Y) * j))

    def evaluate(self, X, Y):
        """Evaluate Func"""
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)  # Convert probabilities to binary predictions

        return predictions, self.cost(Y, A)



    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
