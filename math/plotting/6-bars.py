#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Complete the following source code to plot a stacked bar graph:

fruit is a matrix representing the number of fruit various people possess
The columns of fruit represent the number of fruit Farrah, Fred, and Felicia have, respectively
The rows of fruit represent the number of apples, bananas, oranges, and peaches, respectively
The bars should represent the number of fruit each person possesses:
The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
Each fruit should be represented by a specific color:
apples = red
bananas = yellow
oranges = orange (#ff8000)
peaches = peach (#ffe5b4)
A legend should be used to indicate which fruit is represented by each color
The bars should be stacked in the same order as the rows of fruit, from bottom to top
The bars should have a width of 0.5
The y-axis should be labeled Quantity of Fruit
The y-axis should range from 0 to 80 with ticks every 10 units
The title should be Number of Fruit per Person
"""

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]
labels = ['Farrah', 'Fred', 'Felicia']
plt.ylim(0, 80)
plt.yticks(range(0, 81, 10))
plt.title("Number of Fruit per Person")
plt.ylabel("Quantity of Fruit")
plt.bar(labels, fruit[0][:], color='red', label='apples', width=.5)
plt.bar(labels, fruit[1][:], color='yellow', bottom=apples, label='bananas',
        width=.5)
plt.bar(labels, fruit[2][:], color='#FF8000', bottom=apples + bananas,
        label='oranges', width=.5)
plt.bar(labels, fruit[3][:], color='#FFE5B4',
        bottom=apples + bananas + oranges, label='peaches', width=.5)
plt.legend()
plt.show()
