#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal

np.random.seed(5)
X = np.random.multivariate_normal([5, -4, 2], [[6, -3, 5], [-3, 10, -2], [5, -2, 5]], 10000).T
mn = MultiNormal(X)
try:
    mn.pdf(np.array([[1], [2], [3], [4]]))
except ValueError as e:
    print(str(e))
try:
    mn.pdf(np.array([[1, 1], [2, 2], [3, 3]]))
except ValueError as e:
    print(str(e))
