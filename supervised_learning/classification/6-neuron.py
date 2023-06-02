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
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]

        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        pred, cost = self.evaluate(X, Y)

        return pred, cost

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
