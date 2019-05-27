import numpy as np
from numba import njit


@njit
def create_diff_matrix(num_features):
    D = np.full((num_features, num_features), fill_value=-1.0, dtype=np.float_)
    for i in range(num_features):
        D[i, i] = num_features - 1.0
    return D


@njit
def create_ridge_matrix(num_features):
    D = np.eye(num_features)
    return D


# @njit
def multi_class_error(predicted_choices, true_choices, num_choices=None):
    if num_choices is None:
        num_choices = np.sum(true_choices)
    if num_choices < 1.0:
        print("Num choices is not at least 1 !!")
        print(num_choices)
    return np.sum(predicted_choices != true_choices) / 2 / num_choices


@njit
def numba_unique(arr):
    return np.unique(arr)


@njit
def last_argmin(arr):
    return len(arr) - 1 - np.argmin(arr[::-1])


@njit
def last_argmax(arr):
    return len(arr) - 1 - np.argmax(arr[::-1])
