from django.shortcuts import render
from django.http import HttpResponse
import matplotlib.pyplot as plt
import mpld3
from .stock_main import *

# from strategy_comparison.Stocks.analytics.main import graph

# Create your views here.
def homepage(request):
    return render(request, 'main/home.html')


def stock_plot(request):
    fig = graph() # make this stock figure

    html_str = mpld3.fig_to_html(fig)

    return render(request, 'main/home.html', context={'graph': html_str})