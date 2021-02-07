from django.shortcuts import render
from .stock_main import *




# Create your views here.
def homepage(request):
    return render(request, 'main/home.html')

def stock_plot(request):
    graph() # make this stock figure
    fig1 = open("main/graphs/returns.html", 'r').read()
    fig2 = open("main/graphs/graph.html", 'r').read()

    # fig = file_html(fig, CDN, "stock")
    return render(request, 'main/home.html', context={'graph': fig1, 'returns':fig2})

