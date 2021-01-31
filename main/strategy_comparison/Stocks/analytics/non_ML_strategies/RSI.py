import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# function weather to buy or sell an asset

SELL_THRESHOLD = 70
BUY_THRESHOLD = 30

def buy_sell(df):
    orders = []
    flag = -1

    for i in range(0, len(df) - 1):

        # Buy
        if df['RSI'][i] < BUY_THRESHOLD:
            if flag != 1:
                orders.append(-df['Adj Close'][i])
                flag = 1
            else:
                orders.append(np.nan)

        # Sell
        elif df['RSI'][i] > SELL_THRESHOLD and flag != -1:  # We can only sell once we've baught
            if flag != 0:
                orders.append(df['Adj Close'][i])
                flag = 0
            else:
                orders.append(np.nan)
        else:
            orders.append(np.nan)

    # If we end on a buy then sell
    if flag == 1:
        orders.append(df['Adj Close'][-1])
    else:
        orders.append(np.nan)

    return orders


def RSI(df):
    delta = df['Adj Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    
    df['RSI'] = 100 - (100/(1 + rs))
    
    # Skip first 14 days to have real values
    df['RSI'][:14] = 50.

    ##
    df['RSI (buy/sell)'] = buy_sell(df)

    return df


if __name__ == '__main__':
    RSI(pd.read_csv('../SPY.csv', parse_dates=True, index_col=0))

    '''JUNK
    
    MYBE NOT JUNK
        try:
        percent_return = (((buys[0] + total) / buys[0]) * 100) - 100
    except IndexError:
        print('No transactions mamde')
        percent_return = 0
    
    
    # print(df)
    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.get_xaxis().set_visible(False)
    # fig.suptitle('Twitter')
    #
    # df['Close'].plot(ax=ax1)
    # ax1.set_ylabel('Price ($)')
    # df['RSI'].plot(ax=ax2)
    # ax2.set_ylim(0,100)
    # ax2.axhline(30, color='r', linestyle='--')
    # ax2.axhline(70, color='r', linestyle='--')
    # ax2.set_ylabel('RSI')
    #
    # plt.show()'''