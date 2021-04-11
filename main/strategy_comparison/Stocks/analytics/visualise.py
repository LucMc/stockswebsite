import numpy as np
import itertools
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Dark2_5 as palette
from bokeh.palettes import YlGn3 as buy_palette
from bokeh.palettes import YlOrRd4 as sell_palette
from bokeh.themes import built_in_themes
from bokeh.io import curdoc
from bokeh.models import DatetimeTickFormatter

def percent_return(df, col):
    percent_return = (np.sum(df[f'{col} (buy/sell)']) / df['Adj Close'][0]) * 100

    return percent_return

def visualise(df, ticks=[]):
    '''
    Might move this to main somehow, would be nice to get an updated dataframe through each indicator then make the graph
    in main
    '''
    # df = df.head()
    b_colours = itertools.cycle(buy_palette)
    s_colours = itertools.cycle(sell_palette)

    curdoc().theme = 'dark_minimal'
    output_file('graph.html')

    # Subplot MACD
    # MACD = df['MACD Line']
    # signal = df['Signal Line']
    p = figure(
        title='Stock Analytics',
        x_axis_label='Time (Days)',
        y_axis_label='Price ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )


    # ax.title.set_text('IBM Adjusted Close')
    # Make this so it isnt random?
    for tick, buy_colour, sell_colour in zip(ticks, b_colours, s_colours):

        buys = np.where(df[f'{tick} (buy/sell)'] > 0, np.nan, df[f'{tick} (buy/sell)']).__neg__()
        sells = np.where(df[f'{tick} (buy/sell)'] < 0, np.nan, df[f'{tick} (buy/sell)'])
        # if len(buys[np.logical_not(np.isnan(buys))]) != 0:
        p.scatter(df.index, buys, color=buy_colour, legend_label=f'{tick} Buy', marker='^', alpha=1, size=10)
        p.scatter(df.index, sells, color=sell_colour, legend_label=f'{tick} Sell', marker='v', alpha=1, size=10)

    # 1 - Graph of Adj Close price
    p.line(df.index, df['Adj Close'], alpha=0.35)

    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.xaxis.formatter = DatetimeTickFormatter(months=["%d %b"])
    save(p, filename="main/graphs/graph.html")
    return p

def visualise_returns(df, strats=[]):
    output_file('returns.html')

    p = figure(
        title='Cumulative Returns',
        x_axis_label='Time (days)',
        y_axis_label='Return ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'

    )

    df['bh Adj Close'] = df['Adj Close'] - df['Adj Close'][0]

    l = [f'{s} cumulative return' for s in strats]
    l.append('bh Adj Close')

    cm = itertools.cycle(palette)

    for col, colour in zip(l, cm):
        if col == 'bh Adj Close':
            p.line(df.index, df[col], color=colour, legend_label=col, alpha=0.8)
        else:
            p.line(df.index, df[col], color=colour, legend_label=col, line_width=4, alpha=0.8)

    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.xaxis.formatter = DatetimeTickFormatter(months=["%d %b"])

    save(p, filename="main/graphs/returns.html")
    return p

def visualise_MACD(df):
    output_file('MACD.html') # is this needed?

    MACD = df['MACD Line']
    signal = df['Signal Line']

    p = figure(
        title='MACD',
        x_axis_label='Time (Days)',
        y_axis_label='Signal',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )

    p.line(df.index, MACD, legend_label='MACD Line', color='red', line_width=2)
    p.line(df.index, signal, legend_label='Signal Line', color='blue', line_width=2)
    p.xaxis.formatter = DatetimeTickFormatter(months=["%d %b"])

    save(p, filename="main/graphs/MACD.html")
    return p

def visualise_RSI(df):
    output_file('RSI.html') # is this needed?

    p = figure(
        title='RSI',
        x_axis_label='Time (Days)',
        y_axis_label='RSI',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )
    # ax3.set_ylim(0,100)
    p.line(df.index, [30]*len(df.index), color='azure', line_dash='dashed')
    p.line(df.index, [70]*len(df.index), color='azure', line_dash='dashed')

    p.line(df.index, df['RSI'], legend_label='RSI', color='mediumpurple')
    p.xaxis.formatter = DatetimeTickFormatter(months=["%d %b"])

    save(p, filename="main/graphs/RSI.html")
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