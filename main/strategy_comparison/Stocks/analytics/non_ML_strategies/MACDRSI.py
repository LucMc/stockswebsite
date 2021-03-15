'''
This strategy combines the MACD and RSI strategies
'''

import pandas as pd
import numpy as np

# Lower threshold to increase volume
SELL_THRESHOLD = 50 # 56
BUY_THRESHOLD = 50 # 54

def buy_sell(df):
    orders = []
    flag = -1

    for i in range(0, len(df) - 1):

        # Buy
        if df['MACD'][i] > 0 and df['RSI'][i] < BUY_THRESHOLD:
            if flag != 1:
                orders.append(-df['Adj Close'][i])
                flag = 1
            else:
                orders.append(np.nan)

        # Sell
        elif df['MACD'][i] < 0 and flag != -1 and df['RSI'][i] > SELL_THRESHOLD:  # We can only sell once we've baught
            if flag != 0:
                orders.append(df['Adj Close'][i])
                flag = 0
            else:
                orders.append(np.nan)
        else:
            orders.append(np.nan)

    # If we end on a buy then sell
    if flag == 1:
        if flag != 0:
            orders.append(df['Adj Close'][-1])
            flag = 0
        else:
            orders.append(np.nan)
    else:
        orders.append(np.nan)

    return orders

def MACDRSI(df):
    # Create a buy and sell column
    df['MACDRSI (buy/sell)'] = buy_sell(df)
    return df