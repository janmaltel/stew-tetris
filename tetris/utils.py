import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


@njit
def one_hot_vector(one_index, length):
    out = np.zeros(length)
    out[one_index] = 1.
    return out


@njit
def vert_one_hot(one_index, length):
    out = np.zeros((length, 1))
    out[one_index] = 1.
    return out


@njit
def compute_action_probabilities(action_features, weights, temperature):
    utilities = action_features.dot(weights) / temperature
    utilities = utilities - np.max(utilities)
    exp_utilities = np.exp(utilities)
    probabilities = exp_utilities / np.sum(exp_utilities)
    return probabilities


@njit
def grad_of_log_action_probabilities(features, probabilities, action_index):
    features_of_chosen_action = features[action_index]
    grad = features_of_chosen_action - features.T.dot(probabilities)
    return grad


@njit
def softmax(U):
    ps = np.exp(U - np.max(U))
    ps /= np.sum(ps)
    return ps


def plot_learning_curve(plots_path, test_results, x_axis):
    mean_array = np.mean(test_results, axis=(0, 2))
    median_array = np.median(test_results, axis=(0, 2))
    max_array = np.max(test_results, axis=(0, 2))


    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    ax1.plot(x_axis, median_array, label="median")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance"))
    plt.close()

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, max_array, label="max")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "max_performance"))
    plt.close()


def plot_individual_agent(plots_path, tested_weights, test_results, weights_storage, agent_ix, x_axis):
    feature_names = ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
                     'row_transitions', 'eroded', 'hole_depth']
    # Compute tested_weight paths
    fig1, ax1 = plt.subplots()
    for ix in range(tested_weights.shape[1]):
        ax1.plot(x_axis, tested_weights[:, ix], label=feature_names[ix])
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "weight_paths_tested" + str(agent_ix)))
    plt.close()

    # Compute weights_storage paths
    fig1, ax1 = plt.subplots()
    for ix in range(weights_storage.shape[1]):
        ax1.plot(weights_storage[:, ix], label=feature_names[ix])
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "weight_paths" + str(agent_ix)))
    plt.close()

    # Compute learning curves (mean and median)
    mean_array = np.mean(test_results, axis=1)
    median_array = np.median(test_results, axis=1)
    max_array = np.max(test_results, axis=1)

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    ax1.plot(x_axis, median_array, label="median")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance" + str(agent_ix)))
    plt.close()


def plot_analysis(plots_path, tested_weights, test_results, weights_storage, agent_ix, x_axis):
    feature_names = ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
                     'row_transitions', 'eroded', 'hole_depth']
    # Compute tested_weight paths
    fig1, ax1 = plt.subplots()
    for ix in range(tested_weights.shape[1]):
        ax1.plot(x_axis, tested_weights[:, ix], label=feature_names[ix])
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "weight_paths_tested" + str(agent_ix)))
    plt.close()

    # Compute weights_storage paths
    fig1, ax1 = plt.subplots()
    for ix in range(weights_storage.shape[1]):
        ax1.plot(weights_storage[:, ix], label=feature_names[ix])
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "weight_paths" + str(agent_ix)))
    plt.close()

    # Compute weight distances
    # tested_weights = np.random.normal(size=(4, 8))
    diff_weights = np.diff(tested_weights, axis=0)
    distances = np.sqrt(np.sum(diff_weights ** 2, axis=1))
    fig1, ax1 = plt.subplots()
    ax1.plot(distances, label="l2 distance to previous")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "distances" + str(agent_ix)))
    plt.close()

    # Compute RELATIVE weight distances
    relative_diff_weights = np.diff(tested_weights / np.abs(tested_weights[:, 0][:, np.newaxis]), axis=0)
    distances = np.sqrt(np.sum(relative_diff_weights ** 2, axis=1))
    fig1, ax1 = plt.subplots()
    ax1.plot(distances, label="l2 RELATIVE distance to previous")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "relative_distances" + str(agent_ix)))
    plt.close()

    # Compute learning curves (mean and median)
    mean_array = np.mean(test_results, axis=1)
    median_array = np.median(test_results, axis=1)
    max_array = np.max(test_results, axis=1)

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, mean_array, label="mean")
    ax1.plot(x_axis, median_array, label="median")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "mean_performance" + str(agent_ix)))
    plt.close()

    # Plot and save learning curves.
    fig1, ax1 = plt.subplots()
    ax1.plot(x_axis, max_array, label="max")
    plt.legend()
    fig1.show()
    fig1.savefig(os.path.join(plots_path, "max_performance" + str(agent_ix)))
    plt.close()


