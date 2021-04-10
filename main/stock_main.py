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

# from machine_learning import machine_learning
from .strategy_comparison.Stocks.analytics.ML_strategies.ARIMA import *
from .strategy_comparison.Stocks.analytics.strategy_statistics import *
from .strategy_comparison.Stocks.analytics.visualise import *

from .strategy_comparison.Stocks.analytics.ML_strategies.SVR import *
from .strategy_comparison.Stocks.analytics.ML_strategies.NN import visualise_nn
from .generate_dataframe import *
import asyncio

strats = ['MACD', 'RSI', 'MACDRSI']

pd.set_option('display.max_columns', None)  # Helps for printing columns
# pd.set_option('display.max_rows', None)  # Helps for printing rows
'''
Next might want to plot the cumulative return and some statistics of the strategies.
'''


# def generate_training_data(df):
#     year = 2021
#     # year = dt.datetime(year, 1, 1)
#     years = [decrement_year(dt.datetime(year, 1, 1)) for year in range(2020, 2000, -1)]
#     print(years)
#     train = df.copy()
#     for year in years:
#         train = pd.concat([train,
#                        generate_IBM_dataframe(year)]).copy()
#     return train

import time
async def graph(year, date=238):
    # Asyncio
    start = time.time()
    loop = asyncio.get_event_loop()
    year = dt.datetime(year, 1, 1)
    async_df = loop.create_task(generate_df(year))
    # print(df)

    # Machine Learning
    # df = machine_learning(df)

    # Reduce training data for now
    # train = pd.concat([generate_IBM_dataframe(decrement_year(decrement_year(year))),
    #                    (generate_IBM_dataframe(decrement_year(year)))]).copy()

    async_train = loop.create_task(generate_IBM_dataframe(decrement_year(year)))
    await asyncio.wait([async_train, async_df])

    train = pd.DataFrame(async_train.result())

    df = pd.DataFrame(async_df.result())

    test = df.copy()
    # SVR(train, test)

    loop.create_task(test_arima(train, test, df, date=date))
    visualise_nn(df, date)


    # Plot strategies
    visualise(df, ticks=strats) # ticks for which strategy
    # print(df[['MACD (buy/sell)', 'MACD cumulative return']])
    visualise_returns(df, strats=strats) # change to fig2 once working
    visualise_MACD(df)
    visualise_RSI(df)
    print_statistics(df)
    prepare_df(df)
    # train = generate_training_data(df)

    # print(df.columns)
    # df.to_csv('dataframe.csv')
    end = time.time()
    print(end - start)
    await asyncio.wait([async_train, async_df])
    return df
    # do_ml(df)
    # year = increment_year(year)
    # plt.show()




'''
TODO:
 - Get the average return of the MACD Stratergy
 - Somehow make a nice combined graph
 - Neural network just on prices
 - NEAT just on prices

'''