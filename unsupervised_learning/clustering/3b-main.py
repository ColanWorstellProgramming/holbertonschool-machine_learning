#!/usr/bin/env python3

import numpy as np
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randint(0, 100, (300, 3))
    print(optimum_k(X, kmin=3, kmax=2))
    print(optimum_k(X, kmin=2, kmax=2))