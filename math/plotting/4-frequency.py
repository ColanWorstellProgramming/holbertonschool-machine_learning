#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Complete the following source code to plot a histogram of student scores for a project:

The x-axis should be labeled Grades
The y-axis should be labeled Number of Students
The x-axis should have bins every 10 units
The title should be Project A
The bars should be outlined in black
"""

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here