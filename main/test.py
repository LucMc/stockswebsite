import matplotlib.pyplot as plt
import numpy as np
from random import choice
import mpld3

def visualise():
    colours = ['green', 'red', 'purple', 'yellow', 'white', 'pink', 'orange', 'cyan']

    # Subplot MACD
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)

    ax.scatter([0,1,2,3], [0,1,2,3], color='blue', label='a', marker='^', alpha=1, s=10)
    ax.scatter([0,1,2,3], [1,2,3,4], color='red', label='b', marker='v', alpha=1, s=10)

    mpld3.show()

    return fig
visualise()