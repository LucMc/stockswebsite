from django.shortcuts import render
from .stock_main import *
import asyncio


# Normal homepage if no year is selected
def homepage(request):
    return render(request, 'main/home.html')

# Stock plot
def stock_plot(request):
    if not list(request.GET.items()):
        # Set default values
        year = 2000
        ticker = 'IBM'
        date = 100
    else:
        try:
            ticker = request.GET.get('ticker')
            year = int(request.GET.get('year'))
            date = int(request.GET.get('date'))
        except:
            return render(request, 'main/home.html', context={'ticker' : 'An Error has occured please reenter information'})
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(indicators_graph(year=year,
                                      date=date,
                                      ticker=ticker))
    except KeyError:
        return render(request, 'main/home.html', context={'ticker': f'No data for {ticker} in {year}'})

    graph_fig = open("main/graphs/graph.html", 'r').read()
    returns_fig = open("main/graphs/returns.html", 'r').read()

    MACD_fig = open("main/graphs/MACD.html", 'r').read()
    RSI_fig = open("main/graphs/RSI.html", 'r').read()



    return render(request, 'main/home.html', context={'graph': graph_fig, 'returns':returns_fig,
                                                      'MACD':MACD_fig, 'RSI':RSI_fig,
                                                      'ticker': ticker
                                                      })


# Machine Learning Forecasts
def forecast_plot(request):
    if not list(request.GET.items()):
        # Set default values
        year = 2000
        ticker = 'IBM'
        date = 100
    else:
        try:
            ticker = request.GET.get('ticker')
            year = int(request.GET.get('year'))
            date = int(request.GET.get('date'))
        except:
            return render(request, 'main/forecast.html', context={'ticker' : 'An Error has occured please reenter information'})
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(forecast_graph(year=year,
                                      date=date,
                                      ticker=ticker))
    except KeyError:
        return render(request, 'main/forecast.html', context={'ticker': f'No data for {ticker} in {year}'})

    graph_fig = open("main/graphs/graph.html", 'r').read()
    # returns_fig = open("main/graphs/returns.html", 'r').read()

    # MACD_fig = open("main/graphs/MACD.html", 'r').read()
    # RSI_fig = open("main/graphs/RSI.html", 'r').read()

    ARIMA_fig = open("main/graphs/ARIMA.html", 'r').read()
    NN_fig = open("main/graphs/NN.html", 'r').read()


    return render(request, 'main/forecast.html', context={'graph': graph_fig,
                                                      'ARIMA': ARIMA_fig,
                                                      'NN': NN_fig,
                                                      'ticker': ticker
                                                      })