#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class NeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, nodes):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """Forward Propogation"""
        self.__A1 = self.sigmoid(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = self.sigmoid(np.matmul(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def sigmoid(self, X):
        """Sigmoid Helper"""
        return 1 / (1 + np.exp(-X))

    def cost(self, Y, A):
        """Cost Func"""
        m = Y.shape[1]
        j = np.log(1.0000001 - A)
        return ((-1/m) * np.sum(Y * np.log(A) + (1 - Y) * j))

    def evaluate(self, X, Y):
        """Evaluate Func"""
        A = self.forward_prop(X)[1]
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, self.cost(Y, A)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2)

        dZ = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W1 -= alpha * dW
        self.__b1 -= alpha * db
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

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
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        pred, cost = self.evaluate(X, Y)

        return pred, cost

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2
