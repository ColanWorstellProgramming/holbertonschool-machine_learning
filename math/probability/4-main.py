#!/usr/bin/env python3
import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(4)
lam = np.random.uniform(0.5, 10.0)
n = np.random.randint(100, 1000)
data = np.random.exponential(1 / lam, n).tolist()
e = Exponential(data)
x = np.random.randint(1, 100)
print(np.format_float_scientific(e.pdf(x), precision=10))
x = np.random.uniform(1.0, 100.0)
print(np.around(e.pdf(x), 10))
print(np.around(e.pdf(0), 10))
