# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:02:02 2021

@author: Marissa Murphy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('filename.csv')

indepVar = data['section1'].values
depVar = data['section2'].values

meanX = np.mean(indepVar)
meanY = np.mean(depVar)

n = len(indepVar)

# Using the formula to calculate 'm' and 'c'
numer = 0
denom = 0
for i in range(n):
    numer += (indepVar[i] - meanX) * (depVar[i] - meanY)
    denom += (indepVar[i] - meanX) ** 2
    m = numer / denom
    c = meanY - (m * meanX)
 
# Printing coefficients
print("Coefficients: ")
print(m, c)

# Plotting Values and Regression Line
 
maxX = np.max(indepVar) + 100
minX = np.min(indepVar) - 100
 
# Calculating line values x and y
x = np.linspace(minX, maxX, 1000)
y = c + m * x
 
# Plot line

# Or plot with GNU plot?

plt.plot(x, y, color='r', label='Regression Line')
# Ploting Scatter Points
plt.scatter(indepVar, depVar, c='b', label='Scatter Plot')
plt.show()

