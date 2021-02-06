import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from random import choice
from bokeh.plotting import figure, output_file, show, save
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.palettes import Dark2_5 as palette

import itertools

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
    colours = itertools.cycle(palette)
    curdoc().theme = 'dark_minimal'
    output_file('graph.html')

    # Subplot MACD
    # MACD = df['MACD Line']
    # signal = df['Signal Line']
    p = figure(
        title='IBM Stock Analytics',
        x_axis_label='X Axis',
        y_axis_label='Y Axis',
        sizing_mode='scale_width',
        height=200
    )


    # ax.title.set_text('IBM Adjusted Close')
    # Make this so it isnt random?
    for tick, buy_colour, sell_colour in zip(ticks, colours, colours):

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
    save(p, filename="main/graphs/returns.html")
    return p

def visualise_returns(df, strats=[]):
    output_file('graph.html')

    p = figure(
        title='Cumulative Returns',
        x_axis_label='X Axis',
        y_axis_label='Y Axis',
        sizing_mode='scale_width',
        height=200
    )

    df['delta Adj Close'] = df['Adj Close'] - df['Adj Close'][0]
    l = [f'{s} cumulative return' for s in strats]
    l.append('delta Adj Close')

    cm = itertools.cycle(palette)

    for col, colour in zip(l, cm):
        p.line(df.index, df[col], color=colour, legend_label=col)
    # df[l].plot(cmap='RdGy', title='Cumulative Return ($)', legend='best')
    # plt.show()
    save(p, filename="main/graphs/graph.html")
    return p

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