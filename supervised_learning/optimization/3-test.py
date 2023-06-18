#!/usr/bin/env python3

import numpy as np

train_mini_batch = __import__('3-mini_batch').train_mini_batch

def one_hot(Y, classes):
    """convert an array to a one-hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh

# set variables
np.random.seed(3)
s1, s2 = np.random.randint(15, 50, 2)
b1, b2 = np.random.randint(100, 200, 2)
b = np.random.randint(4, 10)
e1 = s1 + (b1 * (2 ** b))
e2 = s2 + (b2 * (2 ** b))
c = 10
lib= np.load('./MNIST.npz')
X_train = lib['X_train'][s1:e1].reshape((e1 - s1, -1))
Y_train = one_hot(lib['Y_train'][s1:e1], c)
X_valid = lib['X_valid'][s2:e2].reshape((e2 - s2, -1))
Y_valid = one_hot(lib['Y_valid'][s2:e2], c)
train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=(2 ** b), epochs=1, load_path='./evaluate.chpt', save_path='./test.chpt')
