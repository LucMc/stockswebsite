'''
This Stratagy used the Moving Average Convergence/Divergence (MACD) crossover to determin when to buy and sell

Acheived higher return by having 100% SLI and not budgeting trades from a total pot (e.g. only use 1% of money per trade
just incase). However since the strategy uses moving averages it manages to mitigate the impacts of market crashes.

TODO:
 - Improve the graph quality, using Tkinter for the GUI or some more advanced matplotlib to show the MACD.
 - If the idea is to get this on a website then consider live streaming via sockets.
'''

import pandas as pd
import numpy as np


# function weather to buy or sell an asset

def buy_sell(signal):
    orders = []
    flag = -1

    for i in range(0, len(signal) - 1):

        # Buy
        if signal['MACD'][i] > 0:
            if flag != 1:
                orders.append(-signal['Adj Close'][i])
                flag = 1
            else:
                orders.append(np.nan)

        # Sell
        elif signal['MACD'][i] < 0 and flag != -1:  # We can only sell once we've baught
            if flag != 0:
                orders.append(signal['Adj Close'][i])
                flag = 0
            else:
                orders.append(np.nan)
        else:
            orders.append(np.nan)

    # If we end on a buy then sell
    if flag == 1:
        if flag != 0:
            orders.append(signal['Adj Close'][-1])
            flag = 0
        else:
            orders.append(np.nan)
    else:
        orders.append(np.nan)

    return orders



def MACD(df):

    # Calculate the MACD and signal line indicators
    # Short term exponential moving average (EMA)
    ShortEMA = df['Adj Close'].ewm(span=12, adjust=False).mean()

    # Calculate long term ema
    LongEMA = df['Adj Close'].ewm(span=26, adjust=False).mean()

    # Calculate MACD
    MACD = ShortEMA - LongEMA

    # Calculate the signal line
    signal = MACD.ewm(span=9, adjust=False).mean()



    # Create new columns for the data
    df['MACD Line'] = MACD
    df['Signal Line'] = signal
    df['MACD'] = MACD - signal

    # Create a buy and sell column
    df['MACD (buy/sell)'] = buy_sell(df)

    # sell shares at the end
    if df['MACD (buy/sell)'].dropna()[-1] < 0:
        df['MACD (buy/sell)'][-1] = df['Adj Close'][-1]

    # buys = [x for x in a[0] if str(x) != 'nan']
    # sells = [x for x in a[1] if str(x) != 'nan']
    #
    #
    # total = sum(sells) - sum(buys) # total return
    # # print(df)
    # if len(buys) > len(sells):
    #     total += df['Adj Close'][-1] # add final closing price if we have an outstanding baught share

    # df.drop(['MACD Line', 'Signal Line'], axis=1, inplace=True)
    return df

if __name__ == '__main__':
    MACD(pd.read_csv('../SPY.csv', parse_dates=True, index_col=0))