import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque

import os
import numpy as np
import pandas as pd
import random
import time
import datetime as dt
import sys
from main.generate_dataframe import generate_df

from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Dark2_5 as palette
from bokeh.palettes import YlGn3 as buy_palette
from bokeh.palettes import YlOrRd4 as sell_palette
from bokeh.themes import built_in_themes
from bokeh.io import curdoc
import itertools


pd.set_option('display.max_columns', None)  # Helps for printing columns
pd.set_option('display.max_rows', None)  # Helps for printing rows
'''
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
'''


def create_model(X_train):
    model = Sequential()
    model.add(Dense(1028, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(7))
    model.add(Activation("linear"))

    return model

def gen_training_data(df=None, year=None):
    if year != None:
        df = generate_df(year).copy()

    # print(df)
    days = 7
    y_train = []
    # print(df[['Adj Close', 'delta Adj Close', 'delta 1d']])
    # print([f'delta {i}d' for i in range(days)])
    for index, row in df.iterrows():
        y_train.append(row[[f'delta {i+1}d' for i in range(days)]].to_list())

    y_train = np.array(y_train[1:-7])
    X_train = df[['Volume', 'delta', 'MACD Line', 'Signal Line', 'MACD']].to_numpy()[1:-7]

    return X_train, y_train


def NN(df=None):
    # df = pd.read_csv('dataframe.csv')
    year = 2000
    year = dt.datetime(year, 1, 1)

    training_data = gen_training_data(year)
    X_train = training_data[0]
    y_train = training_data[1]
    # print(X_train[0])
    # sys.exit()

    # print(df)
    print(X_train.shape, y_train.shape)
    model = create_model(X_train)
    # optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])

    # Train on multiple years
    for year in range(2001, 2020):
        year = dt.datetime(year, 1, 1)

        training_data = gen_training_data(year)
        X_train = training_data[0]
        y_train = training_data[1]
        model.fit(X_train, y_train, batch_size=2, epochs=20)
        model.save("NN.model")
        # print(model.predict([60.0]))
        # time.sleep(1)

    print(model.predict(X_train))
    print(y_train)

    # d = {'Adj Close': df['Adj Close'],
    #      'delta 1': (df['Adj Close'].shift(-1) - df['Adj Close'])/df['Adj Close']}
    # print(pd.DataFrame(d))
    # print(df['Adj Close'].shift(-1))

def visualise_nn(df, date):
    # year = 2020 # this would be any given year
    # date = 130
    # year = dt.datetime(year, 1, 1)

    testing_data = gen_training_data(df)
    X_test = testing_data[0]
    y_test = testing_data[1]

    test = np.array(X_test[date])
    test = test.reshape(1,len(X_test[0]))
    print(y_test)
    model = tf.keras.models.load_model("main/models/NN.model")
    prediction = model.predict(test)
    print("Prediction:", prediction)
    print("actual:", y_test[date])

    # Recalculate values based on delta
    print(y_test[date][0])

    # df = generate_df(year).copy()
    print("actual:", y_test[date])
    adjclose = df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values
    print((y_test[date]*adjclose) + adjclose)
    predictions = (prediction*adjclose) + adjclose
    # print(predictions)
    print(f"DATE: {date}")

    generate_graph(predictions, df, date)
    # return predictions

def generate_graph(fc, df, date):
    days = 7
    cm = itertools.cycle(palette)

    curdoc().theme = 'dark_minimal'
    output_file('NN.html')
    p = figure(
        title='IBM NN Learning',
        x_axis_label='Time (Days)',
        y_axis_label='Price ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )
    print([df.index[x-1] for x in range(date, date+days)])
    print("Forecast:", fc[0])
    print("Actual:",df['Adj Close'][date:date+days])

    p.line([df.index[x] for x in range(date, date+days)], fc[0], alpha=0.35, color='orange', line_width=4)
    # forecast = p.line([df.index[x-1] for x in range(i, i+step)], fcs[i], alpha=0.35, color='orange', radius=1)
    p.line(df.index[date:date+days], df['Adj Close'][date:date+days], alpha=0.35, color=cm.__next__(), line_width=4)

    # create widget and link
    # slider = Slider(start=0, end=255, step=1, value=10)
    # slider.js_link('value', forecast.glyph, 'radius')
    #
    # show(column(forecast, slider))

    save(p, filename="main/graphs/NN.html")

if __name__ == '__main__':
    visualise_nn(2020, 150)
    # NN()

    # print(list(df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values))
    # print(df)
    # print(len(df), len(X_test))
    # print(list(df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values))

