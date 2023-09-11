#!/usr/bin/env python3

likelihood = __import__('0-likelihood').likelihood
import numpy as np

try:
    likelihood(1, '50', np.linspace(0, 1, 11))
except ValueError as e:
    print(str(e))
try:
    likelihood(1, -5, np.linspace(0, 1, 11))
except ValueError as e:
    print(str(e))
try:
    likelihood(0, 0, np.linspace(0, 1, 11))
except ValueError as e:
    print(str(e))
