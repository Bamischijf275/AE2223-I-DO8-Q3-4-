import matplotlib.pyplot as plt
import numpy as np


def generateResultsChart(SD1, SD2, SD3, WS):

    labels = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    SD1bar = ax.bar(x - (3*width/2), SD1, width, label='StarDist1')
    SD2bar = ax.bar(x - width/2, SD2, width, label='StarDist2')
    SD3bar = ax.bar(x + width/2, SD3, width, label='StarDist3')
    WSbar = ax.bar(x + (3*width/2), WS, width, label='Watershed')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value [-]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(SD1bar, padding=3)
    ax.bar_label(SD2bar, padding=3)
    ax.bar_label(SD3bar, padding=3)
    ax.bar_label(WSbar, padding=3)


    fig.tight_layout()

    plt.ylim(0.4, 1)
    plt.savefig("EffectivenessOfDifferentMethods")
