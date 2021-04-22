import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

import numpy as np
import pandas as pd
import datetime as dt
from main.generate_dataframe import generate_df

from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import DatetimeTickFormatter
from bokeh.io import curdoc
import itertools

import time
from tensorflow.keras.callbacks import TensorBoard
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
    model.add(Dense(7, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


# def create_model(X_train, layer_size=128, num_of_layers=3):
#     model = Sequential()
#     model.add(Dense(layer_size, input_shape=X_train.shape[1:]))
#     model.add(Activation('relu'))
#     for _ in range(num_of_layers):
#         model.add(Dense(layer_size))
#         model.add(Activation('relu'))
#
#     # model.add(Dense(64))
#     # model.add(Activation('relu'))
#
#     model.add(Dense(7))
#     model.add(Activation("linear"))
#
#     return model

def gen_training_data(df=None, year=None):
    ticker = "IBM"
    if year != None:
        df = generate_df(year, ticker).copy()

    # print(df)
    days = 7
    y_train = []
    # print(df[['Adj Close', 'delta Adj Close', 'delta 1d']])
    # print([f'delta {i}d' for i in range(days)])
    for index, row in df.iterrows():
        y_train.append(row[[f'delta {i+1}d' for i in range(days)]].to_list())

    y_train = np.array(y_train[1:-7])
    X_train = df[['delta', 'MACD Line', 'Signal Line', 'MACD']].to_numpy()[1:-7]

    return X_train, y_train

def optimise_network(X_train,y_train,
                     X_train_total, y_train_total):
    # Testing parameters
    layer_sizes = [1048, 523, 128, 64, 32, 16, 8, 4]
    batch_sizes = [64, 32, 16, 8, 4, 2]
    num_of_layers = [6, 5, 4, 3, 2, 1]

    random.shuffle(layer_sizes)
    random.shuffle(batch_sizes)
    random.shuffle(num_of_layers)


    for layer_size in layer_sizes:
        for batch_size in batch_sizes:
            for num_layers in num_of_layers:

                NAME = f"NN-{layer_size}-{batch_size}-{num_layers}-{int(time.time())}"
                tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

                model = create_model(X_train, layer_size=layer_size, num_of_layers=num_layers)
                # optimizer = tf.keras.optimizers.Adam(0.001)
                model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
                model.fit(X_train_total, y_train_total, batch_size=batch_size, epochs=40, callbacks=[tensorboard])


def NN(df=None):
    # df = pd.read_csv('dataframe.csv')
    year = 2000
    year = dt.datetime(year, 1, 1)

    training_data = gen_training_data(year=year)
    X_train = training_data[0]
    y_train = training_data[1]
    # print(X_train[0])
    # sys.exit()

    # print(df)

    X_train_total = X_train
    y_train_total = y_train

    # Train on multiple years
    for year in range(2002, 2020): # Change to 2000
        year = dt.datetime(year, 1, 1)

        training_data = gen_training_data(year=year)
        # X_train_total += training_data[0]
        for x in training_data[0]:
            np.append(X_train_total, x)
        for y in training_data[1]:
            np.append(y_train_total, y)
        # y_train_total.append(y)
    print(X_train_total.shape, y_train.shape)

    # optimise_network(X_train,y_train,
    #                  X_train_total, y_train_total)

    NAME = f"Final-DNN-{int(time.time())}"
    tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

    model = create_model(4, 4)
    # optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    print(X_train_total[0])

    # reshape for LSTM
    X_train_total = X_train_total.reshape(-1, len(X_train_total), 4)
    y_train_total = y_train_total.reshape(-1, len(y_train_total), 7)
    X_train = X_train.reshape(-1, len(X_train), 4)

    model.fit(X_train_total, y_train_total, batch_size=1, epochs=25, callbacks=[tensorboard])
    print(y_train_total)

    # BEST LOSS 4-64-4
    # BEST ACCURACY 4-2-2
    model.save("../../../../../main/models/LSTM.model")
    print("Prediction:",model.predict(X_train))
    print("Actual:",y_train)

    # d = {'Adj Close': df['Adj Close'],
    #      'delta 1': (df['Adj Close'].shift(-1) - df['Adj Close'])/df['Adj Close']}
    # print(pd.DataFrame(d))
    # print(df['Adj Close'].shift(-1))

async def visualise_nn(df, date):
    # year = 2020 # this would be any given year
    # date = 130
    # year = dt.datetime(year, 1, 1)

    testing_data = gen_training_data(df)
    X_test = testing_data[0]
    y_test = testing_data[1]

    test = np.array(X_test[date-4:date])
    print(test)
    test = test.reshape(-1, 4, 4)

    # print(y_test)
    model = tf.keras.models.load_model("main/models/LSTM.model")
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

    generate_graph(predictions, df, date)
    # return predictions

def generate_graph(fc, df, date):
    days = 7
    cm = itertools.cycle(palette)

    curdoc().theme = 'dark_minimal'
    output_file('NN.html')
    p = figure(
        title='LSTM Forecast',
        x_axis_label='Time (Days)',
        y_axis_label='Price ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )
    print([df.index[x-1] for x in range(date, date+days)])
    print("Forecast:", fc[0])
    print("Actual:",df['Adj Close'][date:date+days].values)

    p.line([df.index[x] for x in range(date, date+days)], fc[0], alpha=0.35, color=cm.__next__(),
           line_width=4, legend_label="Forecast")
    # forecast = p.line([df.index[x-1] for x in range(i, i+step)], fcs[i], alpha=0.35, color='orange', radius=1)
    p.line(df.index[date:date+days], df['Adj Close'][date:date+days].values, alpha=0.35, color=cm.__next__(),
           line_width=4, legend_label="Stock Price")

    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    # create widget and link
    # slider = Slider(start=0, end=255, step=1, value=10)
    # slider.js_link('value', forecast.glyph, 'radius')
    #
    # show(column(forecast, slider))
    p.xaxis.formatter = DatetimeTickFormatter(days=["%d %b"])

    save(p, filename="main/graphs/NN.html")

if __name__ == '__main__':
    # visualise_nn(2020, 150)
    NN()

    # print(list(df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values))
    # print(df)
    # print(len(df), len(X_test))
    # print(list(df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values))

