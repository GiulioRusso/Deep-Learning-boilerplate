from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt


def ROC_AUC_plot(figsize: Tuple[int, int],
                 title: str,
                 experiment_ID: str,
                 ticks: List[int],
                 epochs_ticks: np.ndarray,
                 ROC_AUC: List[float],
                 ROC_AUC_path: str):
    """
    ROC AUC plot

    :param figsize: figure size
    :param title: plot title
    :param experiment_ID: experiment ID
    :param ticks: ticks
    :param epochs_ticks: epochs ticks
    :param ROC_AUC: ROC_AUC
    :param ROC_AUC_path: ROC AUC path
    """

    # Figure: ROC AUC
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontweight="bold", fontsize=18, y=1.0)
    plt.title("{}".format(experiment_ID), style='italic', fontsize=10, pad=10)
    plt.grid()
    plt.plot(ticks, ROC_AUC, marker=".", color='blue')
    plt.xlabel("Epochs")
    plt.xticks(epochs_ticks)
    plt.ylabel("ROC AUC")
    plt.ylim(0.0, 1.0)
    plt.savefig(ROC_AUC_path, bbox_inches='tight')
    plt.clf()  # clear figure
    plt.close(fig)
