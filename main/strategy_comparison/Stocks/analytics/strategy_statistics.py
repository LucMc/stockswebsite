'''
This file was used to give statistical information for strategic analysis.

'''
import numpy as np
def sucessful_trades(df, col):

    # buys = [x for x in df[f'Buy_{col}'] if str(x) != 'nan']
    # sells = [x for x in df[f'Sell_{col}'] if str(x) != 'nan']
    profitable_trades = 0
    loss_trades = 0

    # Test this is working
    buys = df[df[f'{col} (buy/sell)'] < 0][f'{col} (buy/sell)'].__neg__()
    sells = df[df[f'{col} (buy/sell)'] > 0][f'{col} (buy/sell)']

    for i in range(len(buys)):
        if buys[i] < sells[i]:
            profitable_trades += 1
        else:
            loss_trades += 1


    return profitable_trades, loss_trades


def percent_return(df, col):
    percent_return = (np.sum(df[f'{col} (buy/sell)']) / df['Adj Close'][0]) * 100

    return percent_return

def average_return(df):
    '''
    line of best fit to achieve average return
    '''
    from statistics import mean
    import numpy as np

    #ys = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    ys = df['Adj Close']
    xs = np.array([i for i in range(len(ys))], dtype=np.float64)

    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    # b for plotting the graph
    # b = mean(ys) - m*mean(xs)

    return m*100

def cumulative_returns(df, strats=[]):
    '''
    So this does the cumulative return I think, just not to many dp and also needs to fill in the rest of the rows..
    '''
    for s in strats:
        cumulative_return = 0
        current_trade = 0
        data = df[f'{s} (buy/sell)'].fillna(0)
        df[f'{s} cumulative return'] = 0.
        for i in range(len(data)):
            if data[i] < 0:
                current_trade = data[i]
            elif data[i] > 0:
                cumulative_return += data[i] + current_trade

            df.at[data.index[i], f'{s} cumulative return'] = cumulative_return

    return df

def print_statistics(df):
    '''
    Method to return all the statistics so that it doesn't have to be done from main.
    Maybe add a rolling average to show the performance over time. Maybe cretate a results
    dataframe.
    '''
    # average return if just held for a year by average gradient
    holding_return = ((df['Adj Close'][-1] - df['Adj Close'][0])/ df['Adj Close'][0]) * 100
    print(f"Average Year\'s Return {average_return(df)}%")
    print(f"Buy and hold Return {holding_return}%\n")

    for strat in ['MACD', 'RSI', 'MACDRSI']:
        print(f"{strat} Year\'s Return {percent_return(df, strat)}")
        print(f"{strat} Sucess rate {sucessful_trades(df, strat)}")
        print('\n')
