#!/usr/bin/env python3

likelihood = __import__('0-likelihood').likelihood
import numpy as np

try:
    likelihood(25, 20, np.linspace(0, 1, 11))
except ValueError as e:
    print(str(e))
likelihood(30, 30, np.linspace(0, 1, 11))