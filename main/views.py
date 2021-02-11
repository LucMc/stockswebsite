from django.shortcuts import render
from .stock_main import *




# Create your views here.
def homepage(request):
    return render(request, 'main/home.html')

def stock_plot(request):
    try:
        graph(int(request.POST['year'])) # make this stock figure
    except Exception as e:
        graph(2000) # make this stock figure

    graph_fig = open("main/graphs/graph.html", 'r').read()
    returns_fig = open("main/graphs/returns.html", 'r').read()

    MACD_fig = open("main/graphs/MACD.html", 'r').read()
    RSI_fig = open("main/graphs/RSI.html", 'r').read()

    return render(request, 'main/home.html', context={'graph': graph_fig, 'returns':returns_fig,
                                                      'MACD':MACD_fig, 'RSI':RSI_fig
                                                       })

