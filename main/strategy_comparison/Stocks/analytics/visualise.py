import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('dark_background')

def percent_return(df, col):
    percent_return = (np.sum(df[f'{col} (buy/sell)']) / df['Adj Close'][0]) * 100

    return percent_return

def visualise(df, ticks=[]):
    '''
    Might move this to main somehow, would be nice to get an updated dataframe through each indicator then make the graph
    in main
    '''
    # Subplot MACD
    MACD = df['MACD Line']
    signal = df['Signal Line']
    fig = plt.figure(figsize=(12.2, 8.5))
    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=4, colspan=1)
    ax2 = plt.subplot2grid((10, 1), (5, 0), rowspan=2, colspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((11, 1), (9, 0), rowspan=2, colspan=1, sharex=ax1)


    ax1.title.set_text('IBM Adjusted Close')
    ax2.title.set_text('MACD')

    # Visually show the stock buy and sell signals
    # The best strategy should be plotted, just using MACD for now
    '''Make these into a loop'''
    if ticks.__contains__('MACD'):
        buys = np.where(df['MACD (buy/sell)'] > 0, np.nan, df['MACD (buy/sell)']).__neg__()
        sells = np.where(df['MACD (buy/sell)'] < 0, np.nan, df['MACD (buy/sell)'])

        ax1.scatter(df.index, buys, color='green', label='MACD Buy', marker='^', alpha=1, s=10) # size of marker param s
        ax1.scatter(df.index, sells, color='red', label='MACD Sell', marker='v', alpha=1, s=10)

    # RSI
    # This validation should be added to others
    if ticks.__contains__('RSI') :
            buys = np.where(df['RSI (buy/sell)'] > 0, np.nan, df['RSI (buy/sell)']).__neg__()
            sells = np.where(df['RSI (buy/sell)'] < 0, np.nan, df['RSI (buy/sell)'])
            print(buys[np.logical_not(np.isnan(buys))])
            if len(buys[np.logical_not(np.isnan(buys))]) != 0:
                ax1.scatter(df.index, buys, color='purple', label='RSI Buy', marker='^', alpha=1, s=10)
                ax1.scatter(df.index, sells, color='yellow', label='RSI Sell', marker='v', alpha=1, s=10)


    # MACDRSI
    if ticks.__contains__('MACDRSI'):
        buys = np.where(df['MACDRSI (buy/sell)'] > 0, np.nan, df['MACDRSI (buy/sell)']).__neg__()
        sells = np.where(df['MACDRSI (buy/sell)'] < 0, np.nan, df['MACDRSI (buy/sell)'])

        ax1.scatter(df.index, buys, color='blue', label='MACDRSI Buy', marker='^', alpha=1, s=25) # size of marker param s
        ax1.scatter(df.index, sells, color='white', label='MACDRSI Sell', marker='v', alpha=1, s=25)

    # 1 - Graph of Adj Close price
    ax1.plot(df['Adj Close'], label='Close Price', alpha=0.35)
    plt.xlabel('Date')
    ax1.set_ylabel('Adj Close Price USD ($)')
    ax1.legend(loc='best')

    # 2 - Graph of MACD indicator
    ax2.plot(df.index, MACD, label='MACD Line', color='red', linewidth=2)
    ax2.plot(df.index, signal, label='Signal Line', color='blue', linewidth=2)
    ax2.legend(loc='upper left')

    # 3 - Graph of RSI indicator
    ax3.title.set_text('RSI')

    ax3.set_ylim(0,100)
    ax3.plot(df.index, [30]*len(df.index), color='r', linestyle='--')
    ax3.plot(df.index, [70]*len(df.index), color='r', linestyle='--')

    ax3.set_ylabel('RSI')

    ax3.plot(df.index, df['RSI'], label='RSI', color='purple')

    # plt.show()
    return fig

def visualise_returns(df, strats=[]):
    df['delta Adj Close'] = df['Adj Close'] - df['Adj Close'][0]
    l = [f'{s} cumulative return' for s in strats]
    l.append('delta Adj Close')
    df[l].plot(cmap='RdGy', title='Cumulative Return ($)', legend='best')
    plt.show()