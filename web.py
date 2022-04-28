import streamlit as st
from datetime import date

import yfinance as yf


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

START = "2015-01-01"
TODAY = "2019-01-01"
#TODAY = date.today().strftime("%Y-%m-%d")
stock = 'ADA-USD'
company = 'ADA-USD'


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(stock)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# how many days do we want to base our predictions on ?
prediction_days = 10

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

def LSTM_model():

    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model

model = LSTM_model()
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5',
                               verbose = 2,
                               save_best_only = True)

model.fit(x_train,
          y_train,
          epochs=25,
          batch_size = 32,
          callbacks = [checkpointer])

START = "2019-01-01"
TODAY = "2022-01-01"
test_data = load_data(stock)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

def plot_prediction():
  plt.plot(actual_prices, color='black', label=f"Actual {company} price")
  plt.plot(predicted_prices, color= 'green', label=f"predicted {company} price")
  plt.title(f"{company} share price")
  plt.xlabel("time")
  plt.ylabel(f"{company} share price")
  plt.legend()
  plt.show()

plot_prediction()

a = plot_prediction()

plt.plot(actual_prices, color='black', label=f"Actual {company} price")
plt.plot(predicted_prices, color= 'green', label=f"predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel("time")
plt.ylabel(f"{company} share price")
plt.legend()
plt.show()

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Forecast :)')


# Plot Prediction
def plot_prediction():
  plt.plot(actual_prices, color='black', label=f"Actual {company} price")
  plt.plot(predicted_prices, color= 'green', label=f"predicted {company} price")
  plt.title(f"{company} share price")
  plt.xlabel("time")
  plt.ylabel(f"{company} share price")
  plt.legend()
  plt.show()


st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(plot_prediction())

#st.write(plot_prediction())
