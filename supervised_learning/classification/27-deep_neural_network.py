#!/usr/bin/env python3
"""Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        """Forward Propagation"""
        A = X
        self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            Z = np.matmul(W, A) + b

            if i == self.__L:
                A = self.softmax(Z)
            else:
                A = self.sigmoid(Z)

            self.__cache['A' + str(i)] = A

        return A, self.__cache

    def softmax(self, X):
        """Softmax Activation Function"""
        expZ = np.exp(X)
        return expZ / np.sum(expZ, axis=0)

    def sigmoid(self, X):
        """Sigmoid Helper"""
        return 1 / (1 + np.exp(-X))

    def evaluate(self, X, Y):
        """Evaluate Func"""
        if Y is None:
            return None

        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=0)

        return predictions, self.cost(Y, A)

    def cost(self, Y, A):
        """Cost Func"""
        if Y is None:
            return None

        m = Y.shape[0]
        cost = -np.sum(np.log(A[Y, np.arange(m)])) / m
        return cost

    def one_hot_decode(self, one_hot):
        """Decode Fun"""
        if type(one_hot) is not np.ndarray:
            return None
        if one_hot.ndim != 2:
            return None
        return np.argmax(one_hot, axis=0)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient Descent"""
        m = Y.shape[1]
        L = self.__L

        A = cache["A" + str(L)]
        dZ = A - Y

        for l in range(L, 0, -1):
            A_prev = cache["A" + str(l - 1)]
            W = self.__weights["W" + str(l)]
            b = self.__weights["b" + str(l)]

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.__weights["W" + str(l)] -= alpha * dW
            self.__weights["b" + str(l)] -= alpha * db

            if l > 1:
                dZ = dA * (A_prev * (1 - A_prev))

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
