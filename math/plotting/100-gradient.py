#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Complete the following source code to create a scatter plot of sampled elevations on a mountain:

The x-axis should be labeled x coordinate (m)
The y-axis should be labeled y coordinate (m)
The title should be Mountain Elevation
A colorbar should be used to display elevation
The colorbar should be labeled elevation (m)
"""

np.random.seed(5)

x = np.random.randn(2000) * 10
y = np.random.randn(2000) * 10
z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

# your code here
