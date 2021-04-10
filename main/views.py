from django.shortcuts import render
from .stock_main import *
import asyncio


# Normal homepage if no year is selected
def homepage(request):
    return render(request, 'main/home.html')

# Stock plot
def stock_plot(request):
    if int(request.GET.get('year')) == 0:
        return render(request, 'main/home.html')
    else:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(graph(int(request.GET.get('year')), int(request.GET.get('date'))))  # make this stock figure

    # print(int(request.GET.get('year')))
    # print("date", int(request.POST['date']))
    # except Exception as e:
    #     print("ERROR", e)
    #     graph(2000) # make this stock figure

    graph_fig = open("main/graphs/graph.html", 'r').read()
    returns_fig = open("main/graphs/returns.html", 'r').read()

    MACD_fig = open("main/graphs/MACD.html", 'r').read()
    RSI_fig = open("main/graphs/RSI.html", 'r').read()

    ARIMA_fig = open("main/graphs/ARIMA.html", 'r').read()
    NN_fig = open("main/graphs/NN.html", 'r').read()


    return render(request, 'main/home.html', context={'graph': graph_fig, 'returns':returns_fig,
                                                      'MACD':MACD_fig, 'RSI':RSI_fig,
                                                      'ARIMA': ARIMA_fig,
                                                      'NN': NN_fig
                                                      })

