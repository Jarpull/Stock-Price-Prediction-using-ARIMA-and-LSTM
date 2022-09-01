# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:29:48 2022

@author: Jarpul
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.api import qqplot

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('C:/Users/Jarpul/Downloads/TLKM.JK.csv',
                parse_dates=['Date'])
df.info()

column = ["Date","Open"]
d1 = df[column]

d1 = df.resample('D', on='Date').sum()[['Open']]
d1

d1.isnull().sum()

fig, ax = plt.subplots(2, 1, figsize=(15, 12))

ax[0].plot(d1)
ax[0].set_title('Before')

# replace 0 values
ts_data_na = d1.replace(0, np.nan)

# fill missing values
ts_data_final = ts_data_na.dropna()

ax[1].plot(ts_data_final)
ax[1].set_title('After')

plt.show()

#plot open price
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Opening Prices')
plt.plot(df['Open'])
plt.title('TLKM Opening price')
plt.show()
plt.close()

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Opening Prices')
plt.plot(ts_data_final)
plt.title('TLKM Opening price')
plt.show()
plt.close()

def test_stationarity(timeseries, extra=''):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation - '+extra)
    plt.gcf().set_size_inches(18,6)
    plt.savefig('ADF-Test-'+extra+'.jpg', bbox_inches='tight')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    #timeseries = timeseries.iloc[:,0].values
    timeseries = timeseries.values
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
test_stationarity(ts_data_final, 'Original')

# ACF & PACF 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,3), dpi= 80)
fig.subplots_adjust(top=0.8)

plot_acf(ts_data_final, ax=ax1, lags=20)
plot_pacf(ts_data_final, ax=ax2, lags=20)

# font size of tick labels
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.suptitle('ACF & PACF')

plt.show()

# 1st order differencing
data_1d = df.Open.diff().dropna()
plt.figure(figsize=(20,7))
plt.plot(data_1d)
plt.title('1st Order Differencing')

# ACF & PACF 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,3), dpi= 80)
fig.subplots_adjust(top=0.8)

plot_acf(data_1d, ax=ax1, lags=20)
plot_pacf(data_1d, ax=ax2, lags=20)

# font size of tick labels
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.suptitle('ACF & PACF')

plt.show()

# 2nd order differencing
data_2d = df.Open.diff().diff().dropna()
plt.figure(figsize=(20,7))
plt.plot(data_2d)
plt.title("2nd Order Differencing")

# ACF & PACF 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,3), dpi= 80)
fig.subplots_adjust(top=0.8)

plot_acf(data_2d, ax=ax1, lags=20)
plot_pacf(data_2d, ax=ax2, lags=20)

# font size of tick labels
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.suptitle('ACF & PACF')

plt.show()

# data train and test 
train_data, test_data = ts_data_final[3:int(len(ts_data_final)*0.8)], ts_data_final[int(len(ts_data_final)*0.8):]
plt.figure(figsize=(14,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Opening Prices')                              
plt.plot(ts_data_final, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()

print(f'Total dataset: {len(ts_data_final)}')
print(f'Train data: {len(train_data)}')
print(f'Test data: {len(test_data)}')

test_data
np.savetxt("testdataTLKM.csv", test_data, '%s')

# 0,1,2 ARIMA Model 1 1 2
model1 = ARIMA(train_data, order=(1,1,2))
fitted = model1.fit()
print(fitted.summary())

fc, se, conf = fitted.forecast(len(test_data), alpha=0.05)  # 95% conf
print(fc)
np.savetxt("Forecasted112.csv", fc, '%.4f')


# Make as pandas series
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)
# Plot
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Prediksi Harga Saham TLKM')
plt.xlabel('Time')
plt.ylabel('Harga Saham TLKM')
plt.legend(loc='upper left', fontsize=8)
plt.show()

plt.figure(figsize=(18,8))
plt.title('Grafik Hasil Prediksi')
plt.xlabel('Time', fontsize=20)
plt.ylabel('Harga Pembukaan (IDR)', fontsize=20)
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(fc_series, label="Forecasted Price")
plt.legend()
plt.show()

fc = fc.reshape(50,1)

mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))

