import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')


start = dt.datetime(2019, 1, 1)
end = dt.datetime(2020, 12, 31)

# Will want to edit these values parsed
df = web.DataReader('SPY', 'yahoo', start, end)
df.to_csv('SPY.csv')


df = pd.read_csv('../Stocks/analytics/SPY.csv', parse_dates=True, index_col=0) # All these options for how you want the df to look
#print(df.head())

print(df[['Open', 'High']].head())

df['Adj Close'].plot()
plt.show()