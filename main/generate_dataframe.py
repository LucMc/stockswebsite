import pandas_datareader.data as web
from .strategy_comparison.Stocks.analytics.non_ML_strategies.MACD import MACD
from .strategy_comparison.Stocks.analytics.non_ML_strategies.RSI import RSI
from .strategy_comparison.Stocks.analytics.non_ML_strategies.MACDRSI import MACDRSI
from .strategy_comparison.Stocks.analytics.ML_strategies.SVM import *
from .strategy_comparison.Stocks.analytics.strategy_statistics import *
from pandas.plotting import register_matplotlib_converters

import asyncio
register_matplotlib_converters()


def increment_year(end):
    try:
        start = end.replace(year=end.year + 1) # Leap year might cause issues
    except ValueError:
        start = end.replace(year=end.year + 1, day=end.day-1)
    return start

def decrement_year(end):
    try:
        start = end.replace(year=end.year - 1) # Leap year might cause issues
    except ValueError:
        start = end.replace(year=end.year - 1, day=end.day-1)
    return start

async def generate_ticker_dataframe(start, ticker):
    '''
    Gather the last years worth of daily ticker data into a dataframe
    '''
    print('- - ' * 15)
    print(f'Generating {ticker} from {start.year}')
    print('- - ' * 15)

    end = increment_year(start)
    df = web.DataReader(ticker, 'yahoo', start, end)
    return df

async def generate_df(year, ticker):
    strats = ['MACD', 'RSI', 'MACDRSI']
    # Variables
    # year = dt.datetime(year, 1, 1)

    # Generate dataframe
    loop = asyncio.get_event_loop()
    df = loop.create_task(generate_ticker_dataframe(year, ticker))
    await df

    df = df.result()
    # Drop columns not in use
    df.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)

    # Non Machine Learning
    df = MACD(df)
    df = RSI(df)
    df = MACDRSI(df)
    cumulative_returns(df, strats=strats)

    prepare_df(df)

    # train = generate_ticker_dataframe(decrement_year(year)).copy()
    # test = df.copy()
    # SVR(train, test)

    return df