import random
import matplotlib.pyplot as plt
import nest
import numpy as np


def calculate_entropy(weight_matrix):
    """
    Calculate the entropy of the weight matrix.
    """
    flattened_weights = weight_matrix.flatten()
    flattened_weights = flattened_weights[flattened_weights != 0]
    probabilities = flattened_weights / np.sum(flattened_weights)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

def calculate_mean(weight_matrix):
    """
    Calculate the mean of the weight matrix.
    """
    mean_weights = np.mean(weight_matrix)
    return mean_weights