import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque

import os
import numpy as np
import pandas as pd
import random

pd.set_option('display.max_columns', None)  # Helps for printing columns
pd.set_option('display.max_rows', None)  # Helps for printing rows

def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

def NN(df=None):
    df = pd.read_csv('dataframe.csv')
    model = create_model(3, 8)
    days = 7
    y_train = []
    # print(df[['Adj Close', 'delta Adj Close', 'delta 1d']])
    # print([f'delta {i}d' for i in range(days)])
    for index, row in df.iterrows():
        y_train.append(row[[f'delta {i+1}d' for i in range(days)]].to_list())

    # [y_train.append([df[f'delta {i+1}d'].to_list()]) for i in range(days)]
    # y_train = np.array(y_train).T
    # Remove NaN rows for training
    y_train = np.array(y_train[1:-7])
    X_train = np.array(df['delta'].to_list())[1:-7]
    print(y_train)

    model.fit(X_train, y_train)
    # model.predict(55.0)
    # print(df[['Adj Close', 'delta']])


    # print(df)
    d = {'Adj Close': df['Adj Close'],
         'delta 1': (df['Adj Close'].shift(-1) - df['Adj Close'])/df['Adj Close']}
    # print(pd.DataFrame(d))
    # print(df['Adj Close'].shift(-1))

if __name__ == '__main__':
    NN()
