# -*- coding: utf-8 -*-
"""Plain Linear Regression on Bitcoin

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AjJda87EzsMafD-NVLdJuS22geaZfwzW
"""

import os
from google.colab import drive
drive.mount('/content/gdrive')
os.chdir('/content/gdrive/MyDrive/MA440')
!pwd 
!ls

#Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# linear regression on a dataset with outliers
from random import random
from random import randint
from random import seed
from numpy import arange
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot

dogecoin = pd.read_csv('coin_Dogecoin.csv')

dogecoin.head(8)

plt.plot(dogecoin['SNo'],dogecoin['Close'])
plt.title('Dogecoin Prices')
plt.xlabel('Time')
plt.ylabel('Price ($)')

dogecoin.dropna(inplace=True)

required_features = ['Open','High','Low','Volume']
output_label = 'Close'

x_train, x_test, y_train, y_test = train_test_split(
dogecoin[required_features],
dogecoin[output_label],
test_size = 0.3
)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Create the model

model = LinearRegression()
model.fit(x_train, y_train)

model.score(x_test,y_test)

# evaluate a model
def evaluate_model(x_train, y_train, model):
	# define model evaluation method
	cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 42)
	# evaluate model
	scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	# force scores to be positive
	return absolute(scores)
 
# evaluate model
results = evaluate_model(x_train, y_train, model)
print('Mean MAE: %.3f (%.3f)' % (mean(results), std(results)))

#predicting future prices with basic linear regression

future_set = dogecoin.shift(periods=20).tail(50)
#future_set = bitcoin.shift(periods=12).tail(50)
prediction = model.predict(future_set[required_features])

plt.figure(figsize = (14, 7))
plt.plot(dogecoin["SNo"][-400:-60], dogecoin["Close"][-400:-60], color='goldenrod', lw=2)
plt.plot(future_set["SNo"], prediction, color='deeppink', lw=2)
plt.title("Dogecoin Future Price Prediction")


plt.xlabel("Time (Days)", size =12)
plt.ylabel("Price (USD)", size =12 )

#Huber Regression
from random import random
from random import randint
from random import seed
from numpy import arange
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot

model2 = HuberRegressor()
model2.fit(x_train,y_train)

model.score(x_test,y_test)
prediction0 = model.predict(x_test)
print(f"r2 Score Of Test Set : {r2_score(y_test, prediction0)}")

# BayesianRidge 

from sklearn.linear_model import BayesianRidge

model2 = BayesianRidge()
model2.fit(x_train, y_train)

model2.score(x_test, y_test)
prediction1 = model2.predict(x_test)
print(f"r2 Score Of Test Set : {r2_score(y_test, prediction2)}")

#Elastic Net Model

from sklearn.linear_model import ElasticNet

model3 = ElasticNet()
model3.fit(x_train, y_train)

model3.score(x_test, y_test)
prediction2 = model3.predict(x_test)
print(f"r2 Score Of Test Set : {r2_score(y_test, prediction2)}")

#Lasso Regression 

from sklearn.linear_model import Lasso

model4 = Lasso()
model4.fit(x_train, y_train)

model4.fit(x_test, y_test)
prediction3 = model4.predict(x_test)
print(f"r2 Score Of Test Set : {r2_score(y_test, prediction3)}")