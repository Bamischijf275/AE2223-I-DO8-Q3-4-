import matplotlib.pyplot as plt
import numpy as np

SD1 = [0.4, 0.5, 0.45, 0.55]
SD2 = [0.42, 0.52, 0.47, 0.57]
SD3 = [0.38, 0.48, 0.43, 0.53]
WS = [0.3, 0.4, 0.4, 0.4]


def generateResultsChart(SD1, SD2, SD3, WS):

    SD1min = [0.1, 0.1, 0.1, 0.1]
    SD1max = [0.2, 0.2, 0.2, 0.2]

    SD2min = [0.1, 0.1, 0.1, 0.1]
    SD2max = [0.1, 0.1, 0.1, 0.1]

    SD3min = [0.1, 0.1, 0.1, 0.1]
    SD3max = [0.1, 0.1, 0.1, 0.1]

    WSmin = [0.1, 0.1, 0.1, 0.1]
    WSmax = [0.1, 0.1, 0.1, 0.1]

    labels = [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']

    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()

    SD1bar = ax.bar(x - (3*width/2), SD1, width, label='StarDist1')
    ax.errorbar(x - (3*width/2), SD1, yerr=[SD1min, SD1max], fmt='ko', capsize=5)

    SD2bar = ax.bar(x - width/2, SD2, width, label='StarDist2')
    ax.errorbar(x - width/2, SD2, yerr=[SD2min, SD2max], fmt='ko', capsize=5)

    SD3bar = ax.bar(x + width/2, SD3, width, label='StarDist3')
    ax.errorbar(x + width/2, SD3, yerr=[SD3min, SD3max], fmt='ko', capsize=5)

    WSbar = ax.bar( x + (3*width/2), WS, width, label='Watershed')
    ax.errorbar(    x + (3*width/2), WS, yerr=[WSmin, WSmax], fmt='ko', capsize=5)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value [-]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.ylim(0.3, 0.8)
    plt.savefig("EffectivenessOfDifferentMethods")


generateResultsChart(SD1, SD2, SD3, WS)
