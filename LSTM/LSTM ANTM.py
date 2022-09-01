# -*- coding: utf-8 -*-
"""
Created on Sat May 21 17:04:25 2022

@author: Jarpul
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import time
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential, load_model
from keras.layers import LSTM 
from keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from keras import callbacks
import os
import math

np.random.seed(0)

df = pd.read_csv('C:/Users/Jarpul/Downloads/ANTM.JK.csv',)

column_list=['Date','Open']
df_final=df[column_list]                                                                         
df_final.head(3)

df_final = df_final.set_index("Date")

df_final.isnull().sum()

# Visualization of stock closing price history
plt.figure(figsize=(18,6))
plt.title('Grafik Riwayat Perubahan Harga Saham')
plt.plot(df_final['Open'])
plt.xticks(range(0,df.shape[0],80),df['Date'].loc[::80])
plt.xlabel('Waktu',fontsize=18)
plt.ylabel('Harga Penutupan (IDR)',fontsize=18)
plt.show()

# Scaling the data with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Transform the data into Scaled Data
scaledData = scaler.fit_transform(df_final)

file = np.asarray(scaledData)
np.savetxt("ScaledDataANTM10064.csv", file, '%s', delimiter=",")

# Split dataset to 80% for training and 20% for testing
trainingDataLen = math.ceil(len(df_final) * 0.8)

# Display the training data length
trainingDataLen

# Convert created dataframe into numpy array
closedataset = df_final.values

# Create a new dataset which contain scaled data value
trainData = scaledData[:trainingDataLen , :]

# Split into trained x and y dataset
xTrain = []
yTrain = []
for i in range(60, len(trainData)):
    xTrain.append(trainData[i-60:i , 0])
    yTrain.append(trainData[i , 0])

    if i<= 61:
        print(xTrain)
        print(yTrain)
        print()

# Convert trained x and y into numpy array
xTrain, yTrain = np.array(xTrain), np.array(yTrain)

# Reshape x trained data from 2 dimension array to 3 dimension array
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
xTrain.shape

# Parameter Initialization
jumlah_units = 50
optimasi     = 'adam'                  
loss_func    = 'mean_squared_error'
ukuran_batch = 64
jumlah_epoch = 100

# Build LSTM model
model = Sequential()

# First LSTM layer
model.add(LSTM(units = jumlah_units, 
               return_sequences=True, 
               input_shape=(xTrain.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units = jumlah_units, 
               return_sequences=False))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(units =1))

# Compile model
model.compile(optimizer= optimasi, loss= loss_func)

# Counting execution time
start_time = time.time()

# Train the model
hist = model.fit(xTrain, yTrain, 
                 batch_size = ukuran_batch, 
                 epochs     = jumlah_epoch)
print("--- %s detik ---" % (time.time() - start_time))

# Counting average loss
np.mean(hist.history['loss'])

# Create testing dataset, new array which contains scaled value
testData = scaledData[trainingDataLen - 60: , :]

# Create dataset xTest and yTest
xTest = []
yTest = closedataset[trainingDataLen: , :]
for i in range(60, len(testData)):
    xTest.append(testData[i - 60:i, 0])

# Convert test set as numpy array
xTest = np.array(xTest)
xTest

# Reshape test set from 2 dimension array to 3 dimension array
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# Get predicted stock price value
predicted = model.predict(xTest)

# Unscaling the predicted value
predictions = scaler.inverse_transform(predicted)

file = np.asarray(predictions)
np.savetxt("PredictionsANTM10064.csv", file, '%s', delimiter=",")

mape = np.mean(np.abs(predictions - yTest)/np.abs(yTest))
print('MAPE: '+str(mape))

# Add prediction for Plot
train = df.loc[:trainingDataLen, ['Date', 'Open'] ]
valid = df.loc[trainingDataLen:, ['Date', 'Open'] ]

# Create dataframe prediction
dfPrediction = pd.DataFrame(predictions, columns = ['predictions'])

# Ploting data to graph train and validation
training = df_final[:trainingDataLen]
validation = df_final[trainingDataLen:]
validation['Predictions'] = predictions

# Visualize full graph
plt.figure(figsize=(18,8))
plt.title('Grafik Hasil Prediksi')
plt.xlabel('Date', fontsize=20)
plt.xticks(range(0,df.shape[0],80),df['Date'].loc[::80])
plt.ylabel('Harga Penutupan (IDR)', fontsize=20)
plt.plot(training['Open'])
plt.plot(validation[['Open', 'Predictions']])
plt.legend(['Training', 'Actual', 'Predictions'], loc='lower right')
plt.show()

# Visualize test and predicted data
plt.figure(figsize=(18,8))
plt.title('Grafik Hasil Prediksi')
plt.xlabel('Date', fontsize=20)
plt.xticks(range(0,df.shape[0],25),df['Date'].loc[::2])
plt.ylabel('Harga Penutupan (IDR)', fontsize=20)
plt.plot(validation[['Open', 'Predictions']])
plt.legend(['Actual', 'Predictions'], loc='lower right')
plt.show()

file = np.asarray(validation)
np.savetxt("ValidationANTM10064.csv", file, '%s', delimiter=",")
