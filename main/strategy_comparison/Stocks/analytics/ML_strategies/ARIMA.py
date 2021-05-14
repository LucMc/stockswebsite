import pandas as pd
from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Dark2_5 as palette

from bokeh.io import curdoc
from bokeh.models import DatetimeTickFormatter

import itertools
from pmdarima.arima import auto_arima

def train_arima(train, test, i, step=7):
    '''Make it so it's a year plus the appended ammount until slider (done by splitting df)'''
    step = 7

    train = pd.concat([train, test[:i]])
    # order = get_order(train)

    result = auto_arima(train['Adj Close'])
    # model = ARIMA(train['Adj Close'], order=get_order(train)) # find these from yt vid
    # result = model.fit(disp=0)
    # print(result.predict(step))
    return result.predict(step)
    # pickle.dump(result, open("main/models/ARIMA.pickle", 'wb'))

async def test_arima(train, test, df, date):
    STEP = 7
    # result = pickle.load(open("main/models/ARIMA.pickle", 'rb'))
    # fc, se, conf = result.forecast(step)
    train.index = pd.DatetimeIndex(train.index).to_period('M')
    test.index = pd.DatetimeIndex(test.index).to_period('M')

    fc = train_arima(train, test, date, step=STEP)
    # ses.append(se)
    # confs.append(conf)

    visualise_arima(fc, df, date, step=STEP)
    '''
    Split the dataset into 2 week blocks to test on?
    or have a slider
    '''
def visualise_arima(fc, df, date, step=7):
    cm = itertools.cycle(palette)

    curdoc().theme = 'dark_minimal'
    output_file('ARIMA.html')
    p = figure(
        title='ARIMA Forecast',
        x_axis_label='Time (Days)',
        y_axis_label='Price ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )

    p.line([df.index[x] for x in range(date, date+step)], fc, alpha=0.35, color=cm.__next__(),
           line_width=4, legend_label="Forecast")
    # forecast = p.line([df.index[x-1] for x in range(i, i+step)], fcs[i], alpha=0.35, color='orange', radius=1)
    p.line(df.index[date:date+7], df['Adj Close'][date:date+7], alpha=0.35, color=cm.__next__(),
           line_width=4, legend_label="Stock Price")

    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    # create widget and link
    # slider = Slider(start=0, end=255, step=1, value=10)
    # slider.js_link('value', forecast.glyph, 'radius')
    #
    # show(column(forecast, slider))
    p.xaxis.formatter = DatetimeTickFormatter(days=["%d %b"])

    save(p, filename="main/graphs/ARIMA.html")

'''
def test_arima(train, test, df):
    STEP = 14
    # result = pickle.load(open("main/models/ARIMA.pickle", 'rb'))
    # fc, se, conf = result.forecast(step)
    train.index = pd.DatetimeIndex(train.index).to_period('M')
    test.index = pd.DatetimeIndex(test.index).to_period('M')

    fcs, ses, confs = [],[],[]

    for i in range(len(test)):
        fc = train_arima(train, test, i, step=STEP)
        fcs.append(fc)
        # ses.append(se)
        # confs.append(conf)

    visualise_arima(fcs, df, step=STEP)
    '''
# Split the dataset into 2 week blocks to test on?
# or have a slider
'''
def visualise_arima(fcs, df, step=14):
    cm = itertools.cycle(palette)

    curdoc().theme = 'dark_minimal'
    output_file('ARIMA.html')
    p = figure(
        title='IBM ARIMA Learning',
        x_axis_label='Time (Days)',
        y_axis_label='Price ($)',
        sizing_mode='scale_width',
        height=200,
        height_policy='max'
    )

    for i, fc in enumerate(fcs[:-12]):
        # i = 10
        p.line([df.index[x-1] for x in range(i, i+step)], fc, alpha=0.35, color='orange')
        forecast = p.circle([df.index[x-1] for x in range(i, i+step)], fcs[i], alpha=0.35, color='orange', radius=1)
    p.line(df.index, df['Adj Close'], alpha=0.35, color=cm.__next__())

    # create widget and link
    # slider = Slider(start=0, end=255, step=1, value=10)
    # slider.js_link('value', forecast.glyph, 'radius')
    #
    # show(column(forecast, slider))

    save(p, filename="main/graphs/ARIMA.html")
'''
'''
TODO:
 - Make slider
 - Add bounds for predictions
 - Find values for order
'''


'''

def arimaold(df):
    train_data, test_data = df[0:int(len(df) * 0.7)], df[int(len(df) * 0.7):]
    training_data = train_data['Adj Close'].values
    test_data = test_data['Adj Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4, 1, 0)) # work out order
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))

    # TESTING

    test_set_range = df[int(len(df) * 0.7):].index
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed', label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title('IBM Prices Prediction')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.xticks()
    plt.legend()
    plt.show()
'''