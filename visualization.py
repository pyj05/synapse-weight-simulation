import random
import matplotlib.pyplot as plt
import nest
import numpy as np


def plot_diff_matrix(ax, w_matrix, init_matrix, title_str=""):
    """
    Draw a heatmap of (w_matrix - init_matrix).
    """
    cax = ax.imshow(w_matrix - init_matrix, cmap='jet', vmin=-0.1, vmax=0.5)
    cbarB = plt.colorbar(cax, ax=ax)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xticklabels(["1", "3", "5", "7", "9"])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_yticklabels(["1", "3", "5", "7", "9"])
    ax.set_xlabel("to neuron")
    ax.set_ylabel("from neuron")
    ax.set_title(title_str)

