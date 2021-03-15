import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from .SVM import prepare_df
plt.style.use('fivethirtyeight')


def prepare(df):
    # Get all the data except last row
    # prepare_df()
    df = df.head(len(df)-1)

    # Create empty list for independent and dependent data
    days = list()
    adj_close_prices = list()

    # Get the dates and the adjusted close prices
    df_days = df.index
    df_adj_close = df.loc[:, 'Adj Close']

    # Create the independant data set
    for i in range(len(df_days)):
        days.append([df_days[i].day])

    # Create dependant dataset
    for adj_close_price in df_adj_close:
        adj_close_prices.append([float(adj_close_price)])

    days = [[x] for x in range(len(days))]
    # print("printing days", days)
    # print(adj_close_prices)
    return days, adj_close_prices


def SVR(df, test):
    from sklearn.svm import SVR

    actual_price = df.tail(1)

    # Change to X and y train test etc
    days, adj_close_prices = prepare(df)
    test_days, test_adj_close_prices = prepare(test)


    # Create the 3 Support Vector Regression Models
    # # Linear kernel
    # lin_svr = SVR(kernel='linear', C=1000.0)
    # lin_svr.fit(days, adj_close_prices)
    #
    # # Polynomial kernel
    # poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
    # poly_svr.fit(days, adj_close_prices)

    # rbf kernel
    rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.15)
    rbf_svr.fit(days, adj_close_prices)

    print("prediction: ", rbf_svr.predict([[252]]))

    # Plot models on graph to see which has the best fit
    # plt.figure(figsize=(16,8))
    # plt.scatter(test_days, test_adj_close_prices, color='red', label='Data')
    # plt.plot(test_days, rbf_svr.predict(test_days), color='green', label='RBF Model')
    # plt.plot(test_days, poly_svr.predict(test_days), color='orange', label='Polynomial Model')
    # plt.plot(test_days, lin_svr.predict(test_days), color='blue', label='Linear Model')
    # plt.legend()
    # plt.show()