'''
This is the main file for all machine learning,
strategies of machine learning might get numerous enough to take their own files for which this will be the parent.

In order to get an appropriate reward it should reward based on return rather than accuracy since a 10 % day is better
than 9 small 1% trades, therefore in order for it to learn correctly it's output should be smart.
Also I think train against every S&P company otherwise there isn't going to be enough data.

Also added SPY to SPX_tickers to get working
'''
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from statistics import mean

from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def machine_learning(df):
    return df

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('spx_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df[f'{ticker}_{i}d'] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df[f'{ticker}_target'] = list(map(buy_sell_hold,
                                               df[f'{ticker}_1d'],
                                               df[f'{ticker}_2d'],
                                               df[f'{ticker}_3d'],
                                               df[f'{ticker}_4d'],
                                               df[f'{ticker}_5d'],
                                               df[f'{ticker}_6d'],
                                               df[f'{ticker}_7d'] ))
    vals = df[f'{ticker}_target'].values.tolist()
    str_vals = [str(i) for i in vals]
    print(df)
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df[f'{ticker}_target'].values

    return X,y,df

# print(extract_featuresets("AAPL"))

# MACHINE LEARNING


def do_ml(ticker):
    '''
    Currently model only supports one ticker at a time
    '''

    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25)

    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()
    print()
    return confidence

do_ml('AAPL')
# do_ml('AAPL')
# do_ml('ABT')




# with open("spxtickers.pickle","rb") as f:
#     tickers = pickle.load(f)
#     model = do_ml(tickers)
#
# accuracies = []
# for count,ticker in enumerate(tickers):
#
#     if count%10==0:
#         print(count)
#
#     accuracy = do_ml(ticker)
#     accuracies.append(accuracy)
#     try:
#         print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))
#     except:
#         pass