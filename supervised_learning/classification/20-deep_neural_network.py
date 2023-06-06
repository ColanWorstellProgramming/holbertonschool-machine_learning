#!/usr/bin/env python3
"""Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1 or False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                j = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(i + 1)] = j
            else:
                jjj = np.sqrt(2 / layers[i-1])
                jj = np.random.randn(layers[i], layers[i-1]) * jjj
                self.__weights['W' + str(i + 1)] = jj
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """Forward Propogation"""
        A = X
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            Z = np.matmul(W, A) + b
            A = self.sigmoid(Z)
            self.__cache['A' + str(i)] = A

        return A, self.__cache

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
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)

        return predictions, self.cost(Y, A)

    @property
    def L(self):
        """layer getter"""
        return self.__L

    @property
    def cache(self):
        '''itermed val getter'''
        return self.__cache

    @property
    def weights(self):
        '''weight getter'''
        return self.__weights
