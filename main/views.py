from django.shortcuts import render
from .stock_main import *




# Create your views here.
def homepage(request):
    return render(request, 'main/home.html')

def stock_plot(request):
    graph(int(request.GET.get('year')), int(request.GET.get('date'))) # make this stock figure
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

    return render(request, 'main/home.html', context={'graph': graph_fig, 'returns':returns_fig,
                                                      'MACD':MACD_fig, 'RSI':RSI_fig,
                                                      'ARIMA': ARIMA_fig
                                                       })

