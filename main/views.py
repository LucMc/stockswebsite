from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
import mpld3
from .stock_main import *
from bokeh.client import push_session
from threading import Thread
from bokeh.server.server import Server
from bokeh.embed import server_document
from tornado.ioloop import IOLoop
# from strategy_comparison_local.Stocks.analytics.main import graph
from bokeh.embed import components
from bokeh.resources import CDN



# Create your views here.
def homepage(request):
    return render(request, 'main/home.html')

def stock_plot(request):
    graph() # make this stock figure
    fig1 = open("main/graphs/graph.html", 'r').read()
    fig2 = open("main/graphs/returns.html", 'r').read()

    # fig = file_html(fig, CDN, "stock")
    return render(request, 'main/home.html', context={'graph': fig1, 'returns':fig2})

