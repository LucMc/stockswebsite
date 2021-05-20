import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import DatetimeTickFormatter
from bokeh.io import curdoc
import itertools
from bokeh.palettes import YlGn3 as buy_palette
from bokeh.palettes import YlOrRd4 as sell_palette



pd.set_option('display.max_columns', None)  # Helps for printing columns
pd.set_option('display.max_rows', None)  # Helps for printing rows

def buy_hold_sell(predictions, df):
    # threshold = 2 # This would be adjustable based on trader frequency and commission
    # 0 sell 1 buy 2 hold
    orders = [-df['Adj Close'][0]]
    flag = 1
    for i, pred in enumerate(predictions[1:-1]): # pred is mean of next 7 days

        if pred > 0 and flag == 0: # buy if forecast mean increase in price
            orders.append(-df['Adj Close'][i+1])
            flag = 1
        elif flag == 1 and pred < 0: # sell if forecast mean increase in price
            orders.append(df['Adj Close'][i+1])
            flag = 0
        else:
            orders.append(np.nan)

    # sell on last day
    if flag == 1:
        orders.append(df['Adj Close'][len(df)-1])
    else:
        orders.append(np.nan)

    return orders


def visualise_classifier_nn(df, gen_graph=True):
    # df = pd.read_csv('dataframe.csv')
    days = 7
    # model = tf.keras.models.load_model("../../../../../main/models/oldLSTM.model")

    # model = tf.keras.models.load_model("main/models/NEWLSTM-32-2-0.3.model")
    model = keras.models.load_model("main/models/LSTM.model")

    df.drop(df.index[-7:0], inplace=True)
    # df['RSI'] /= 100

    # NOT USING RSI AS IT DECREASED ACCURACY
    data = df[['delta', 'MACD Line', 'Signal Line', 'MACD']].to_numpy() # Model input
    # print(df.head(3))
    data = data.reshape(-1, 1, 4)
    # print(data[0])
    # print(df.head(1)[['delta', 'MACD Line', 'Signal Line', 'MACD', 'RSI']])

    # data = data.reshape(int(len(data)), 1, 4)

    # print(len(data))

    predictions = model.predict(data)

    # Only get the next day
    # predictions = np.delete(predictions, np.s_[1:], axis=1)

    # Get average for the next week
    avr_pred = []
    # print(predictions[:10])
    for row in predictions:
        avr_pred.append(np.mean(row))
    # print(avr_pred[:10])

    # print(predictions)
    # print(df['delta 1d'])

    orders = buy_hold_sell(avr_pred, df)
    # print(orders[:10])
    df['LSTM (buy/sell)'] = orders
    # print(df['LSTM (buy/sell)'])

    if gen_graph == True:
        generate_graph(orders, df)
    return df


def generate_graph(orders, df):

    curdoc().theme = 'dark_minimal'
    output_file('main/graphs/NN_classifier.html')
    p = figure(
        title='LSTM Classification',
        x_axis_label='Time (Days)',
        y_axis_label='Price ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )

    # Buy and Sell ticks
    orders = np.array(orders)

    # Make more efficient
    buys = orders
    sells = -orders

    buys[buys >= 0] = np.nan
    sells[sells >= 0] = np.nan

    buys = -buys
    sells = -sells

    # Stock price
    p.line(df.index, df['Adj Close'], alpha=0.35, line_width=4)

    # Ticks
    p.scatter(df.index, buys, color=buy_palette[0], legend_label=f'Buy', marker='^', alpha=1, size=10)
    p.scatter(df.index, sells, color=sell_palette[0], legend_label=f'Sell', marker='v', alpha=1, size=10)

    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    # create widget and link
    # slider = Slider(start=0, end=255, step=1, value=10)
    # slider.js_link('value', forecast.glyph, 'radius')
    #
    # show(column(forecast, slider))
    p.xaxis.formatter = DatetimeTickFormatter(days=["%d %b"])

    save(p, filename="main/graphs/NN_Classifier.html")

if __name__ == '__main__':
    visualise_classifier_nn()


    # print(list(df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values))
    # print(df)
    # print(len(df), len(X_test))
    # print(list(df['Adj Close'].loc[df["delta 1d"] == y_test[date][0]].values))

