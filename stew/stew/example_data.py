import numpy as np
import stew.utils


def discrete_choice_example_data(num_states=100, num_choices=5, num_features=3, probabilistic=True):
    beta = np.random.normal(loc=0, scale=9, size=num_features)
    print("True beta", beta)
    features = np.random.normal(loc=0, scale=1, size=(num_states*num_choices, num_features)) * 2 ** np.arange(num_features)
    utilities = features.dot(beta)
    states = np.repeat(np.arange(num_states), num_choices)
    choices = np.zeros(num_states * num_choices)
    data = np.hstack((np.zeros((num_states*num_choices, 2)), features))
    data[:, 0] = states
    for st in range(num_states):
        ixs = np.where(states == st)[0]
        utils_st = utilities[ixs]

        if probabilistic:
            utils_st = utils_st - np.max(utils_st)
            exp_utils = np.exp(utils_st)
            probs = exp_utils / np.sum(exp_utils)
            # print(probs)
            choice = np.random.choice(np.arange(num_choices), size=1, p=probs)
            choices[ixs[choice]] = 1.0
        else:
            # Deterministic
            choices[ixs[utils_st == np.max(utils_st)]] = 1.0
    data[:, 1] = choices
    return data


def regression_example_data(num_samples=100, num_features=3):
    D = stew.utils.create_diff_matrix(num_features)
    X = np.random.normal(size=(num_samples, num_features))
    true_beta = np.random.normal(size=num_features)
    eps = np.random.normal(loc=0, scale=0.1, size=num_samples)
    y = true_beta.T.dot(X.T) + eps
    return X, y
