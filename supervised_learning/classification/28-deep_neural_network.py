#!/usr/bin/env python3
"""Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Neural Network Class"""
    def __init__(self, nx, layers, activation='sig'):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1 or False in (np.array(layers) > 0):
            raise TypeError("layers must be a list of positive integers")

        if activation not in ["sig", "tanh"]:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            self.__weights['W' + str(i+1)] = np.random.randn(
                layers[i], nx) * np.sqrt(2/nx)
            self.__weights['b' + str(i+1)] = np.zeros((layers[i], 1))
            nx = layers[i]

    def forward_prop(self, X):
        """Forward Propogation"""

        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            B = self.__weights['b' + str(i)]
            A = self.__cache['A' + str(i - 1)]
            Z = np.dot(self.__weights['W' + str(i)], A) + B
            if i == self.__L:
                self.__cache['A' + str(i)] = self.softmax(Z)
            else:
                if self.__activation == "sig":
                    self.__cache['A' + str(i)] = self.sigmoid(Z)
                elif self.__activation == "tanh":
                    self.__cache['A' + str(i)] = np.tanh(Z)

        for key, value in self.__cache.items():
            self.__cache[key] = np.round(value, 10)

        return self.__cache["A{}".format(self.__L)], self.__cache

    def sigmoid(self, X):
        """Sigmoid Helper"""
        return 1 / (1 + np.exp(-X))

    def softmax(self, Z):
        """Softmax activation function"""
        exps = np.exp(Z - np.max(Z))
        return exps / np.sum(exps, axis=0)

    def cost(self, Y, A):
        """Cost Func"""

        m = Y.shape[1]

        cost = (-1 / m) * np.sum(Y * np.log(A))
        cost = np.round(cost, 10)
        return cost

    def evaluate(self, X, Y):
        """Evaluate Func"""

        A, _ = self.forward_prop(X)
        predictions = np.where(A == np.amax(A, axis=0), 1, 0)

        return predictions, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        for i in reversed(range(1, self.__L + 1)):
            A = cache['A' + str(i)]
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            if i == self.__L:
                dz = A - Y
            else:
                if self.__activation == 'sig':
                    dz = da * self.sigmoid(A)
                elif self.__activation == 'tanh':
                    dz = da * (1 - A**2)
            dw = np.dot(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                da = np.dot(W.T, dz)
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        Xgrp = []
        Ygrp = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            if verbose:
                if i == 0 or i % step == 0:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))

            if graph:
                if i == 0 or i % step == 0:
                    current_cost = self.cost(Y, A)
                    Ygrp.append(current_cost)
                    Xgrp.append(i)
                plt.plot(Xgrp, Ygrp)
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                plt.title('Training Cost')

            if verbose or graph:
                if type(step) is not int:
                    raise TypeError("step must be in integer")
                if step <= 0 or step > iterations:
                    raise ValueError("step must be positive and <= iterations")
        if graph:
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """save a file"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        "load a file"
        try:
            if not filename.endswith('.pkl'):
                filename += '.pkl'

            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception:
            return None

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
