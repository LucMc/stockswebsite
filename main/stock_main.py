'''
Main file which runs all the strategies and compares them.
Components:
 - Stratagies
 - GUI
 _ Executing trades

So this main file generates the dataframe then theres multiple other files one for each stratergy which then return
a result back here.

combining all these strategies in order to create a reliable algorithm, then run it on loads of stocks at the same time
so every day it says what to buy/ sell whatever.

Maybe use a fuzzy value from each indicator and if they all up over 1 then buy. So this would mean if the RSI is really
high but the MACD is lagging behind then you should still buy.

Should also discuss the percentage of successful trades since although RSI only becomes a buy or sell a few times a year
it is often correct when it does predict a buy. SLI and other informative information.

Combine Buy_MACD and Sell_MACD columns so that a buy is a -255.54 and a sell is a +288.88 so it's easy to calculate and
means less columns.

Get seperate dataframes for different things since the main one is getting a bit big.
'''
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
from .strategy_comparison.Stocks.analytics.non_ML_strategies.MACD import MACD
from .strategy_comparison.Stocks.analytics.non_ML_strategies.RSI import RSI
from .strategy_comparison.Stocks.analytics.non_ML_strategies.MACDRSI import MACDRSI
# from machine_learning import machine_learning
from .strategy_comparison.Stocks.analytics.ML_strategies.SVM import *

from .strategy_comparison.Stocks.analytics.strategy_statistics import *
from .strategy_comparison.Stocks.analytics.visualise import *

pd.set_option('display.max_columns', None)  # Helps for printing columns
# pd.set_option('display.max_rows', None)  # Helps for printing rows
'''
Next might want to plot the cumulative return and some statistics of the strategies.
'''
def increment_year(end):
    try:
        start = end.replace(year=end.year + 1) # Leap year might cause issues
    except ValueError:
        start = end.replace(year=end.year + 1, day=end.day-1)
    return start


def generate_SPY_dataframe(start):
    '''
    Gather the last years worth of daily ticker data into a dataframe
    '''
    print('- - ' * 15)
    print(f'Generating Data from {start.year}')
    print('- - ' * 15)

    end = increment_year(start)
    df = web.DataReader('IBM', 'yahoo', start, end)
    return df

def graph(year):
    # Variables
    strats = ['MACD', 'RSI', 'MACDRSI']
    year = dt.datetime(year, 1, 1)

    # Generate dataframe
    df = generate_SPY_dataframe(year)

    # Drop columns not in use
    df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

    # Non Machine Learning
    df = MACD(df)
    df = RSI(df)
    df = MACDRSI(df)
    # print(df)

    # Machine Learning
    # Remove unwanted labels before machine learning
    # df = machine_learning(df)


    # Plot strategies
    visualise(df, ticks=strats) # ticks for which strategy
    cumulative_returns(df, strats=strats)
    print(df)
    visualise_returns(df, strats=strats) # change to fig2 once working
    visualise_MACD(df)
    visualise_RSI(df)
    # print_statistics(df)
    # prepare_df(df)
    # do_ml(df)
    # year = increment_year(year)
    # plt.show()
    #


if __name__ == '__main__':
    graph()


'''
TODO:
 - Get the average return of the MACD Stratergy
 - Somehow make a nice combined graph
 - Neural network just on prices
 - NEAT just on prices

'''