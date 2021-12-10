# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:20:22 2021

@author: Min Sun Kim
"""

# This code was written by Min Sun Kim, a student at East Central University
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Input csv data file. This differs based on coin
data = pd.read_csv("Dogecoin_data.csv")

data.head()

data.tail()

data.describe()

data.shape

data.columns

data.nunique()

data['Marketcap'].unique()

data.isnull().sum()

crypto = 'DOGE'
currency = 'USD'

start = dt.datetime(2015,1,1)
end = dt.datetime.now()

# Updated data is read from the Yahoo finance website
data = web.DataReader(f"{crypto}-{currency}", "yahoo", start, end)

data.index = pd.to_datetime(data.index)

data

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Number of days into the future to predict
pred_days = 30

x_train, y_train = [], []

# Train the model 
for x in range(pred_days, len(scaled_data)):
  x_train.append(scaled_data[x-pred_days:x, 0])
  y_train.append(scaled_data[x, 0])
  
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30, batch_size=32)

test_start = dt.datetime(2021,6,1)
test_end = dt.datetime.now()

test_data = web.DataReader(f"{crypto}-{currency}", "yahoo", test_start, test_end)

test_data.index = pd.to_datetime(test_data.index)

test_data

actual_prices = test_data['Close']

actual_prices = np.array(actual_prices)
actual_prices

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
total_dataset.index = pd.to_datetime(total_dataset.index)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-pred_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

# Test the model
for x in range(pred_days, len(model_inputs)):
  x_test.append(model_inputs[x-pred_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# Plot the data
plt.plot(actual_prices, color='red', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Prediction Prices')
plt.title(f"{crypto}-{currency} Price Predictor")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()
