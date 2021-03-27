from sklearn import svm
import pickle
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
'''
Train on the last year then test on the next.
'''
# Move this to machine learning file after
def prepare_df(df, ml='SVM'):
    hm_days = 7
    for i in range(1, hm_days+1):
        # Percentage change
        df[f'delta {i}d'] = (df['Adj Close'].shift(-i) - df['Adj Close']) / df['Adj Close']

    df['delta'] = df['delta 1d'].shift(1)
    df['delta'].fillna(0, inplace=True)
    return df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(df):
    df[f'target'] = list(map(buy_sell_hold,
                                               df[f'delta 1d'],
                                               df[f'delta 2d'],
                                               df[f'delta 3d'],
                                               df[f'delta 4d'],
                                               df[f'delta 5d'],
                                               df[f'delta 6d'],
                                               df[f'delta 7d']))
    vals = df[f'target'].values.tolist()
    str_vals = [str(i) for i in vals]
    print(df)
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df['Adj Close'].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df[f'target'].values
    X = X.reshape(len(X), 1)
    y = y.reshape(len(y), 1)

    return X,y,df


def do_ml(ticker):
    '''
    Currently model only supports one ticker at a time, should also train on multiple years data and using RSI/MACD
    '''

    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25)

    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])

    clf.fit(X_train, y_train.ravel())
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))
    print()
    print()
    return confidence



##
def train_SVM(df):
    clf = svm.SVR()
    clf.fit(df.drop('Adj Closes').tolist(), df['Adj Close'].tolist())
    pickle.dump(clf)

def test_SVM(df):
    return df