import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import matplotlib.dates as mdates


style.use('ggplot')

df = pd.read_csv('../Stocks/analytics/SPY.csv', parse_dates=True, index_col=0) # All these options for how you want the df to look
#print(df.head())

# print(df[['Open', 'High']].head())
df.reset_index(inplace=True)

print(df.index[0])
print(f"Opening price({df.index[-1]}): {df['Adj Close'][-1]}, Price at end({df.index[0]}): {df['Adj Close'][0]}, market increase: {(df['Adj Close'][-1] / df['Adj Close'][0])*100 - 100}")

df['Adj Close'].plot()
# plt.show()