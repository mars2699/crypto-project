# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:14:39 2021

@author: Mason
"""
import seaborn as sns
import pandas as pd
import pytrends
from pytrends.request import TrendReq
from datetime import datetime 
import requests 
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import multiprocessing as mp
from time import time

print("Number of processors: ", mp.cpu_count())
pytrend = TrendReq()

KEYWORDS = ['BitCoin', 'Dogecoin']
KEYWORDS_CODES = [pytrend.suggestions(keyword = i)[0] for i in KEYWORDS]

df_CODES = pd.DataFrame(KEYWORDS_CODES)
df_CODES

EXACT_KEYWORDS = df_CODES['mid'].to_list()
current_Date = datetime.date(datetime.now())

DATE_INTERVAL = '2019-11-01 '+ current_Date.strftime("%Y-%m-%d")

COUNTRY = ["US","GB","CN","RU","JP","GE"]
CATEGORY = 0
SEARCH_TYPE = ''


Individual_EXACT_KEYWORD = list(zip(*[iter(EXACT_KEYWORDS)]*1))
Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]

dicti = {}
i = 1

for Country in COUNTRY:
    for keyword in Individual_EXACT_KEYWORD:
        try:
            pytrend.build_payload(kw_list = keyword,
                              timeframe = DATE_INTERVAL,
                              geo = Country,
                              cat = CATEGORY,
                              gprop = SEARCH_TYPE)
            dicti[i] = pytrend.interest_over_time()
            i+=1
        except requests.exceptions.Timeout:
            print("Timeout occured")

df_trends = pd.concat(dicti, axis = 1)

df_trends.columns = df_trends.columns.droplevel(0)
df_trends = df_trends.drop('isPartial', axis = 1)
df_trends.reset_index(level = 0, inplace = True)
df_trends.columns = ['date', 'Bitcoin-US', 'Dodgecoin-US',
                     'Bitcoin-GB', 'Dodgecoin-GB', 
                     'Bitcoin-CN', 'Dodgecoin-CN',
                     'Bitcoin-RU', 'Dodgecoin-RU',
                     'Bitcoin-JP', 'Dodgecoin-JP',
                     'Bitcoin-GE', 'Dodgecoin-GE']

sns.set(color_codes = True)

dx = df_trends.plot(figsize = (12,8), x = 'date', y =['Bitcoin-US'],
                    kind = "line", title = "Bitcoin Data Google Trends")

dx.set_xlabel('Date')
dx.set_ylabel('Trends Index')
dx.tick_params(axis = 'both', which = 'both', labelsize = 10)

dx = df_trends.plot(figsize = (12,8), x = 'date', y =['Dodgecoin-US'],
                 kind = "line", title = "Dodgecoin Data Google Trends")
dx.set_xlabel('Date')
dx.set_ylabel('Trends Index')
dx.tick_params(axis = 'both', which = 'both', labelsize = 10)

rawData = pd.read_csv("C:\\Users\\Mason\\Desktop\\MA453\\Final Project Code\\Bitcoin_Data.csv")

dates = df_trends['date']

data = rawData[rawData.index == dates[0]]

for i in range(1,len(df_trends)):
    data = data.append(rawData[rawData.index == dates[i]])

data.head()

data.tail()

data.describe()

data.shape

data.columns

data.nunique()

#data['Marketcap'].unique()

data.isnull().sum()

crypto = 'BTC'
currency = 'USD'

start = datetime(2015,1,1)
end = datetime.now()

data = web.DataReader(f"{crypto}-{currency}", "yahoo", start, end)

data.index = pd.to_datetime(data.index)

#data = rawData[rawData.index == dates[0]]

#for i in range(1,len(df_trends)):
#    data = data.append(rawData[rawData.index == dates[i]])

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

pred_days = 30

x_train, y_train = [], []


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

test_start = datetime(2019,11,1)
test_end = datetime.now()

testRaw_data = web.DataReader(f"{crypto}-{currency}", "yahoo", test_start, test_end)

test_data = testRaw_data[testRaw_data.index == dates[0]]

for i in range(1,len(df_trends)):
    test_data = test_data.append(testRaw_data[testRaw_data.index == dates[i]])

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

for x in range(pred_days, len(model_inputs)):
  x_test.append(model_inputs[x-pred_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

largestValActual = actual_prices[0]
largestValPrediction = prediction_prices[0]

for i in range(0, len(actual_prices)):
    if (actual_prices[i] > largestValActual):
        largestValActual = actual_prices[i]
        
for i in range(0, len(actual_prices)):
    if (actual_prices[i] > largestValPrediction):
        largestValPrediction = prediction_prices[i]

#np.true_divide(actual_prices, largestValActual)
#np.true_divide(prediction_prices, largestValPrediction)

predicted_pricesNormal = (prediction_prices / largestValPrediction)*100
actual_pricesNormal = (actual_prices / largestValActual)*100

dodgeVol = np.array(df_trends['Dodgecoin-US'])

bitcoinVol = np.array(df_trends['Bitcoin-US'])

dodgeVolAvg = sum(dodgeVol) / len(dodgeVol)

bitcoinVolAvg = sum(bitcoinVol) / len(bitcoinVol)
'''
Trying to do parallel 

def volChange(volBitcoin):
    changeVol = []
    for i,x in enumerate(volBitcoin):
        if i == 0 or i == (len(volBitcoin)):
            pass
        else:
            pre = volBitcoin[i - 1]
            current = volBitcoin[i]
            if pre > current:
                changeVol.append(current - pre)
            else:
                changeVol.append(current - pre)
    return changeVol
    
pool_obj = mp.Pool()
'''

'''
Bitcoin Run
'''
changeVol = []
for i,x in enumerate(bitcoinVol):
    if i == 0 or i == (len(bitcoinVol)):
        pass
    else:
        pre = bitcoinVol[i - 1]
        current = bitcoinVol[i]
        if pre > current:
            changeVol.append(current - pre)
        else:
            changeVol.append(current - pre)

changeVolArray = np.array(changeVol)
'''

Dodge Coin Run

changeVolDodge = []
for i,x in enumerate(dodgeVol):
    if i == 0 or i == (len(dodgeVol)):
        pass
    else:
        pre = dodgeVol[i - 1]
        current = dodgeVol[i]
        if pre > current:
            changeVolDodge.append(current - pre)
        else:
            changeVolDodge.append(current - pre)

changeVolDodgeArray = np.array(changeVolDodge)
'''

'''
Bitcoin
'''
diffAvg = []
for i in bitcoinVol:
    if (i > bitcoinVolAvg): 
        diffAvg.append(i - bitcoinVolAvg)
        
    else:
        diffAvg.append(bitcoinVolAvg - i)

diffAvgNormal = np.divide(diffAvg, 300)

changeVolArrayNormal = np.divide(changeVolArray, 150)

'''
Dodgecoin
'''
'''
diffDodgeAvg = []
for i in dodgeVol:
    if (i > dodgeVolAvg): 
        diffDodgeAvg.append(i - dodgeVolAvg)
        
    else:
        diffDodgeAvg.append(dodgeVolAvg - i)

diffDodgeAvgNormal = np.divide(diffDodgeAvg, 300)

changeVolDodgeArrayNormal = np.divide(changeVolDodgeArray, 150)
'''
predicted_pricesNormalAdjust = []

for i in bitcoinVol:
    if (i > bitcoinVolAvg):
        for i,x in enumerate(predicted_pricesNormal - 1):
            print(i)
            if predicted_pricesNormal[i] < 70:
                predicted_pricesNormalAdjust[i] = np.multiply((diffAvgNormal[i] + 1), predicted_pricesNormal[i])
                
            else:
                pass
                #predicted_pricesNormalAdjust = np.multiply((diffAvgNormal + 1), predicted_pricesNormal)
    else:
        pass
        #predicted_pricesNormalAdjust = np.multiply((1 - diffAvgNormal), predicted_pricesNormal)
        #predicted_pricesNormalAdjust = np.multiply((changeVolArrayNormal + 1), predicted_pricesNormal)
'''


for i,x in enumerate(predicted_pricesNormal):
    if i == 0 or i == (len(predicted_pricesNormal) - 1):
        pass
    elif(changeVolDodgeArray[i] > 5):
        predicted_pricesNormalAdjust.append(np.multiply((diffDodgeAvgNormal[i] + 1), predicted_pricesNormal[i]))
    elif(changeVolDodgeArray[i] < -5):
        predicted_pricesNormalAdjust.append(np.multiply((changeVolDodgeArrayNormal[i] + 1), predicted_pricesNormal[i]))
    else:
        predicted_pricesNormalAdjust.append(predicted_pricesNormal[i])
 
'''

for i,x in enumerate(predicted_pricesNormal):
    if(predicted_pricesNormal[i] < 70):
        predicted_pricesNormalAdjust.append(np.multiply((diffAvgNormal[i] + 1), predicted_pricesNormal[i]))
    else:
        predicted_pricesNormalAdjust.append(predicted_pricesNormal[i])
#predicted_pricesNormalAdjust = np.multiply((diffAvgNormal + 1), predicted_pricesNormal)
#predicted_pricesNormalAdjust = np.multiply((1 + changeVol), predicted_pricesNormal)
#predicted_pricesNormalAdjust = predicted_pricesNormalAdjust.diagonal()


#for i in range(0,len(dates)):
#    df_NewDates = df_NewDates.append(data[data.index == dates[i]])




plt.plot(bitcoinVol, color ='blue', label='Volitilty')
plt.plot(actual_pricesNormal, color='red', label='Actual Prices')
plt.plot(predicted_pricesNormalAdjust, color='green', label='Prediction Prices')
plt.title(f"{crypto}-{currency} Price Predictor")
plt.xlabel("Number of Data Points")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()

plt.plot(changeVolArrayNormal, color ='blue', label='Volitilty')
plt.xlabel("Number of Data Points")
plt.ylabel("Change In Volitilty")
plt.title("Change in Volitility for Bitcoin")
plt.show()
        
plt.plot(bitcoinVol, color ='blue', label='Volitilty')
plt.plot(actual_pricesNormal, color='red', label='Actual Prices')
plt.title("Bitcoin Volatility and Price")
plt.xlabel("Number of Data Points")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()

plt.plot(actual_pricesNormal, color='red', label='Actual Prices')
plt.plot(predicted_pricesNormal, color='green', label='Prediction Prices')
plt.title("Bitcoin Normal Least Squares Model")
plt.xlabel("Number of Data Points")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()

plt.plot(actual_pricesNormal, color='red', label='Actual Prices')
plt.plot(predicted_pricesNormalAdjust, color='green', label='Prediction Prices with Volatility')
plt.title("Bitcoin Volatility Least Squares Model")
plt.xlabel("Number of Data Points")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()
