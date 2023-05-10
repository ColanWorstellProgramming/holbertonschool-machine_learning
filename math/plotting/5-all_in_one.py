#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Complete the following source code to plot all 5 previous graphs in one figure:

All axis labels and plot titles should have a font size of x-small (to fit nicely in one figure)
The plots should make a 3 x 2 grid
The last plot should take up two column widths (see below)
The title of the figure should be All in One
"""

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure(layout="constrained")
spec = fig.add_gridspec(3, 2)

fig.suptitle("All in One")

one = fig.add_subplot(spec[0, 0])
one.plot(y0, color='red')
one.set_xlim([0, 10])

two = fig.add_subplot(spec[0, 1])
two.scatter(x1, y1, color='magenta', marker='.')
two.set_title("Men's Height vs Weight", fontsize='x-small')
two.set_xlabel('Height (in)', fontsize='x-small')
two.set_ylabel('Weight (lbs)', fontsize='x-small')

three = fig.add_subplot(spec[1, 0])
three.semilogy(x2, y2)
three.set_title("Exponential Decay of C-14", fontsize='x-small')
three.set_xlabel("Time (years)", fontsize='x-small')
three.set_ylabel("Fraction Remaining", fontsize='x-small')
three.set_xlim(0, 28650)

four = fig.add_subplot(spec[1, 1])
four.plot(x3, y31, c="red", linestyle="--", label="C-14")
four.plot(x3, y32, c="green", label="Ra-226")
four.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
four.set_xlabel("Time (years)", fontsize='x-small')
four.set_ylabel("Fraction Remaining", fontsize='x-small')
four.set_xlim(0, 20000)
four.set_ylim(0, 1)
four.legend()

five = fig.add_subplot(spec[2, :])
five.hist(student_grades, facecolor="#428abd", edgecolor="black",
         bins=range(0, 101, 10), linewidth=0.75)
five.set_title("Project A", fontsize='x-small')
five.set_xlabel("Grades", fontsize='x-small')
five.set_ylabel("Number of Students", fontsize='x-small')
five.set_xticks(ticks=range(0, 101, 10))
five.set_xlim(0, 100)
five.set_ylim(0, 30)

plt.show()
