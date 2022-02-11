import numpy as np
import math
import datetime
import pandas as pd
from pandas import read_csv
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

dataframe = pd.read_csv('Basil Input - Basil Data.csv', header=0, usecols=["Experian Catalist Average Pump  Unleaded Petrol"], skip_blank_lines=True)
dataset = dataframe.values
dataset = dataset.astype('float')

dates = pd.read_csv('Basil Input - Basil Data.csv', header=0, usecols=["Date"], skip_blank_lines=True)
dates = np.array(dates).flatten()
frmt = '%Y-%m-%d'
f_dates = [datetime.datetime.strptime(i, frmt).date() for i in dates]
f_dates = [d.strftime("%b '%y") for d in f_dates]
days = np.arange(len(dates))

#normalisation
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#split test and train data
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print( "Training data size: " +str(len(train)), " Testing data size: " +str(len(test)))

#reformat the test and train data into x and y
def new_dataset(dataset, look_back):
    Xdata, Ydata = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i:(i+look_back), 0]
        Xdata.append(a)
        Ydata.append(dataset[i+look_back, 0])
    return np.array(Xdata), np.array(Ydata)

look_back = 50
x_train, y_train = new_dataset(train, look_back)
x_test, y_test = new_dataset(test, look_back)

#set up the neural network
model = Sequential()

#LSTM layers
model.add(LSTM(500, input_dim = 1, activation='relu', return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(100, activation='relu'))
model.add(Dropout(0.1))

#Dense layer
model.add(Dense(1, activation='relu'))
model.add(Dropout(0.1))

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.fit(x_train, y_train, epochs=50, batch_size=1000, validation_data=(x_test, y_test))

train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
final_train_predict = scaler.inverse_transform(train_predict)
final_test_predict = scaler.inverse_transform(test_predict)

fig,ax = plt.subplots()
ax.plot(days, dataframe,'b-')
ax.plot(days[-len(x_test):], final_test_predict,'r-')
ax.set_xticks(days[::300])
ax.set_xticklabels(f_dates[::300],rotation = 45)
ax.set_ylabel('Price (ppl)'); plt.xlabel('Time (days)')
ax.legend(["Actual", "Prediction"], loc="upper right")
plt.title("Unleaded Petrol fuel prices")
plt.show()
