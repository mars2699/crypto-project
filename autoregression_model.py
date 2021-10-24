# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:36:31 2021

@author: Marissa Murphy
"""

from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

# Change to proper filename
theData = read_csv('filename.csv', header=0, index_col=0, squeeze=True)

# Split dataset, if necessary. The filler number used below, 10, represents the last 
# row of data you'd want to take into account
X = theData.values
train, test = X[1:len(X)-10], X[len(X)-10:]

# Train the model
staticmodel = AutoReg(train, lags=29)
staticmodel_fit = staticmodel.fit()
print('Coefficients: %s' % staticmodel_fit.params)

#Make predictions
predictions = staticmodel_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# Plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()