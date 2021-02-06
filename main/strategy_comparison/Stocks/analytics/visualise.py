import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from random import choice
from bokeh.plotting import figure, output_file, show, save
from bokeh.embed import file_html
from bokeh.resources import CDN

from bokeh.themes import built_in_themes
from bokeh.io import curdoc

style.use('dark_background')

def percent_return(df, col):
    percent_return = (np.sum(df[f'{col} (buy/sell)']) / df['Adj Close'][0]) * 100

    return percent_return

def visualise(df, ticks=[]):
    '''
    Might move this to main somehow, would be nice to get an updated dataframe through each indicator then make the graph
    in main
    '''
    # df = df.head()
    colours = ['green', 'red', 'purple', 'yellow', 'white', 'pink', 'orange', 'cyan']
    curdoc().theme = 'dark_minimal'
    output_file('graph.html')

    # Subplot MACD
    # MACD = df['MACD Line']
    # signal = df['Signal Line']
    p = figure(
        title='IBM Stock Analytics',
        x_axis_label='X Axis',
        y_axis_label='Y Axis',
        sizing_mode='scale_width'
    )


    # ax.title.set_text('IBM Adjusted Close')

    for tick in ticks:
        buy_colour = choice(colours)
        colours.remove(buy_colour)
        sell_colour = choice(colours)
        colours.remove(sell_colour)

        buys = np.where(df[f'{tick} (buy/sell)'] > 0, np.nan, df[f'{tick} (buy/sell)']).__neg__()
        sells = np.where(df[f'{tick} (buy/sell)'] < 0, np.nan, df[f'{tick} (buy/sell)'])
        # if len(buys[np.logical_not(np.isnan(buys))]) != 0:
        p.scatter(df.index, buys, color=buy_colour, legend_label=f'{tick} Buy', marker='^', alpha=1, size=10)
        p.scatter(df.index, sells, color=sell_colour, legend_label=f'{tick} Sell', marker='v', alpha=1, size=10)

    # 1 - Graph of Adj Close price
    p.line(df.index, df['Adj Close'], alpha=0.35)
    # plt.xlabel('Date')
    # ax.set_ylabel('Adj Close Price USD ($)')
    # ax.legend(loc='best')
    # show(p)
    save(p, filename="main/graphs/graph.html")
    return p

def visualise_returns(df, strats=[]):
    df['delta Adj Close'] = df['Adj Close'] - df['Adj Close'][0]
    l = [f'{s} cumulative return' for s in strats]
    l.append('delta Adj Close')
    df[l].plot(cmap='RdGy', title='Cumulative Return ($)', legend='best')
    # plt.show()

'''
    # 2 - Graph of MACD indicator
    ax2.plot(df.index, MACD, label='MACD Line', color='red', linewidth=2)
    ax2.plot(df.index, signal, label='Signal Line', color='blue', linewidth=2)
    ax2.legend(loc='upper left')

    # 3 - Graph of RSI indicator
    ax3.title.set_text('RSI')

    ax3.set_ylim(0,100)
    ax3.plot(df.index, [30]*len(df.index), color='r', linestyle='--')
    ax3.plot(df.index, [70]*len(df.index), color='r', linestyle='--')

    ax3.set_ylabel('RSI')

    ax3.plot(df.index, df['RSI'], label='RSI', color='purple')

'''