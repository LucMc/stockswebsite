import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader as web
import pickle
import requests

'''
Might want to add this to git ignore and generate it in some sort of init/ setup function.
'''


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)

    with open("spxtickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    print(tickers)
    return tickers

#save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("spxtickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('../../SPX_tickers'):
        os.makedirs('../SPX_tickers')

    start = dt.datetime(2000, 1, 1) # Update this
    end = dt.datetime(2015, 1, 1)
    for ticker in tickers:
        try:
            print(ticker)
            if not os.path.exists(f'SPX_tickers/{ticker}.csv'):
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.to_csv(f'SPX_tickers/{ticker}.csv')
            else:
                print(f'Already have {ticker}')

        except Exception as e:
            pass


# Gets all the csv data into one csv file
def compile_data():
    with open("spxtickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv(f'SPX_tickers/{ticker}.csv')
            df.set_index('Date', inplace=True)

            df.rename(columns = {'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
        except Exception as e:
            print("ERROR ON", count)

    print(main_df.head())
    main_df.to_csv('spx_joined_closes.csv')

get_data_from_yahoo(True)
compile_data()