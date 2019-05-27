import numpy as np
from numba import njit
import scipy.optimize as optim
import stew.stew.utils
from sklearn.model_selection import GroupKFold


class StewMultinomialLogit:
    def __init__(self, num_features, lambda_min=-6.0, lambda_max=4.0, alpha=0.1,
                 D=None, method="BFGS", max_splits=10, num_lambdas=40,
                 prior_weights=None, one_se_rule=False, verbose=True, nonnegative=False):
        if D is None:
            self.D = stew.stew.utils.create_diff_matrix(num_features=num_features)
        else:
            self.D = D
        self.method = method
        if prior_weights is None:
            self.start_weights = np.random.normal(loc=0, scale=0.1, size=num_features)
        else:
            self.start_weights = prior_weights
        self.max_splits = max_splits
        self.num_lambdas = num_lambdas
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambdas = np.insert(np.logspace(self.lambda_min, self.lambda_max, num=self.num_lambdas-1), 0, 0.0)
        self.verbose = verbose
        self.alpha = alpha
        self.one_se_rule = one_se_rule
        self.nonnegative = nonnegative
        if self.nonnegative:
            self.bounds = optim.Bounds(np.repeat(-0.1, num_features), np.repeat(np.inf, num_features))
            # self.bounds = optim.Bounds(np.repeat(0, num_features), np.repeat(np.inf, num_features))
            self.method = "L-BFGS-B"

    def fit(self, data, lam, start_weights=None, standardize=False):
        if standardize:
            stds_before_standardization = np.std(data[:, 2:], axis=0)
            stds_before_standardization[stds_before_standardization == 0.0] = 1
            standardized_data = data.copy()
            standardized_data[:, 2:] /= stds_before_standardization
        else:
            standardized_data = data

        if start_weights is None:
            start_weights = self.start_weights

        # For OLS, only estimate c-1 parameters, where c is the number of choice sets.
        deleted_features = False
        if lam == 0 and not self.nonnegative:
            # Sum up choices to know how many choice sets there are.
            num_of_choice_sets = np.sum(standardized_data[:, 1])
            # print("num_of_choice_sets: ", num_of_choice_sets)
            num_of_parameters = standardized_data.shape[1] - 2
            diff = num_of_parameters - num_of_choice_sets
            if diff >= 0:
                deleted_features = True
                # Only keep num_of_choice_sets - 1 predictors (+2 columns for choice set indicator and choice)
                standardized_data = standardized_data[:, :int(num_of_choice_sets + 1)]
                start_weights = start_weights[:int(num_of_choice_sets - 1)]

        if self.nonnegative:
            op = optim.minimize(fun=stew_multinomial_logit_ll_and_grad, x0=start_weights,
                                args=(standardized_data, self.D, lam), jac=True, method=self.method,
                                bounds=self.bounds)
        else:
            op = optim.minimize(fun=stew_multinomial_logit_ll_and_grad, x0=start_weights,
                                args=(standardized_data, self.D, lam), jac=True, method=self.method)
        weights = op.x
        if deleted_features:
            tmp_weights = np.zeros(num_of_parameters)
            tmp_weights[:len(weights)] = weights
            weights = tmp_weights

        if standardize:
            print("before re_standardizing weights")
            print(weights)
            print("stds_before_standardization are")
            print(stds_before_standardization)
            weights /= stds_before_standardization
            # print("after re_standardizing weights")
            # print(self.weights)
        return weights

    def predict(self, new_data, weights=None):
        if weights is None:
            raise ValueError("Please provide weights!")
        return stew_multinomial_logit_predict(new_data, weights)

    def predicted_probabilities(self, new_data, weights=None):
        if weights is None:
            raise ValueError("Please provide weights!")
            # weights = self.weights
        return stew_multinomial_logit_predicted_probabilities(new_data, weights)

    def sgd_update(self, weights, data):
        new_weights = weights + self.alpha * single_choice_set_grad(weights, data)
        return new_weights

    def cv_fit(self, data, standardize=False):
        if standardize:
            stds_before_standardization = np.std(data[:, 2:], axis=0)
            stds_before_standardization[stds_before_standardization == 0.0] = 1
            standardized_data = data.copy()
            standardized_data[:, 2:] /= stds_before_standardization
        else:
            standardized_data = data

        num_choices = len(stew.stew.utils.numba_unique(standardized_data[:, 0]))
        num_splits = int(np.minimum(num_choices, self.max_splits))
        kf = GroupKFold(n_splits=num_splits)
        lam_errors = np.full(shape=self.num_lambdas, fill_value=1.)
        lam_sds = np.full(shape=self.num_lambdas, fill_value=0.)
        weights = np.zeros((self.num_lambdas, standardized_data.shape[1] - 2))
        lam_ix = 0
        converged = False
        stop_index = self.num_lambdas
        while lam_ix < self.num_lambdas and not converged:
            lambda_ix = self.lambdas[lam_ix]
            cv_errors = np.zeros(num_splits)
            for cv_ix, (train_index, test_index) in enumerate(kf.split(standardized_data, groups=standardized_data[:, 0])):
                train_data = standardized_data[train_index]  # len(train_data)
                test_data = standardized_data[test_index]  # len(test_data)
                test_choices = test_data[:, 1]
                test_data = np.delete(test_data, 1, 1)
                cv_weights = self.fit(data=train_data, start_weights=self.start_weights, lam=lambda_ix, standardize=False)
                choices_ix = self.predict(test_data, weights=cv_weights)
                cv_errors[cv_ix] = stew.stew.utils.multi_class_error(choices_ix, test_choices)
            lam_errors[lam_ix] = np.mean(cv_errors)
            lam_sds[lam_ix] = np.std(cv_errors)
            # TODO: Put weight calculation outside of loop (only for lambda_min!!)
            weights[lam_ix, :] = self.fit(data=standardized_data, start_weights=self.start_weights, lam=lambda_ix, standardize=False)
            if np.std(weights[lam_ix, :]) < 0.0001:
                # if self.verbose:
                #     print("Converged at index ", lam_ix, " out of ", self.num_lambdas, "lambdas.")  # lambda: ", lambda_ix, ".
                #     # print("Weights are: ", weights[lam_ix, :])
                stop_index = lam_ix + 1
                converged = True
            lam_ix += 1
        # Data set is growing with time, so the other direction is not needed
        # TODO: include number of samples in lambda calculation
        if np.linalg.norm(weights[0, :] - weights[1, :]) < 0.0001:
            print("Warning: the first lambda results in weights equal to lam = 0. "
                  "lambda_min is increased.")
            self.lambda_min += 0.2
            self.lambdas = np.insert(np.logspace(self.lambda_min, self.lambda_max, num=self.num_lambdas - 1), 0, 0.0)
        if lam_ix == self.num_lambdas:
            print("Warning: lambda grid search did not converge to equal weights. "
                  "lambda_max is increased.")
            self.lambda_max += 0.2
            self.lambdas = np.insert(np.logspace(self.lambda_min, self.lambda_max, num=self.num_lambdas - 1), 0, 0.0)
        if stop_index < self.num_lambdas:
            lam_errors = lam_errors[:stop_index]
            lam_sds = lam_sds[:stop_index]
            weights = weights[:stop_index]
        if self.one_se_rule:
            cv_min_ix = stew.stew.utils.last_argmin(lam_errors)
            if self.verbose:
                print("1SE RULE before : Lambda ", cv_min_ix, " out of ", stop_index, "lambdas.")  # is: ", cv_min_lambda, ". I
            cv_min_error = lam_errors[cv_min_ix]
            one_sd_error = cv_min_error + (lam_sds[cv_min_ix] / num_choices) # np.sqrt(num_splits)
            smaller_than_one_sd = lam_errors < one_sd_error
            cv_min_ix = stew.stew.utils.last_argmax(smaller_than_one_sd)
            cv_min_lambda = self.lambdas[cv_min_ix]
            if self.verbose:
                print("1SE RULE after : Lambda ", cv_min_ix, " out of ", stop_index, "lambdas.")  # is: ", cv_min_lambda, ". I
        else:
            cv_min_ix = stew.stew.utils.last_argmin(lam_errors)
            cv_min_lambda = self.lambdas[cv_min_ix]
            if self.verbose:
                print("Lambda min index ", cv_min_ix, " out of ", stop_index, "lambdas.")  # is: ", cv_min_lambda, ". I
            # print("Weights are: ", weights[cv_min_ix, :])

        cv_min_weights = weights[cv_min_ix]
        # print(cv_min_lambda)
        # if self.verbose:
        #     print("lam_errors")
        #     print(lam_errors)
        #     print("weights")
        #     print(weights)
        if standardize:
            # if self.verbose:
            #     print("before re_standardizing weights")
            #     print(cv_min_weights)
            #     print("stds_before_standardization are")
            #     print(stds_before_standardization)
            cv_min_weights /= stds_before_standardization
        return cv_min_weights, cv_min_lambda


@njit
def stew_multinomial_logit_predict(new_data, weights):
    utilities = new_data[:, 1:].dot(weights)
    old_state = new_data[0, 0]
    state_start_row_ix = 0
    choices = np.zeros(len(new_data), np.float_)
    for row_ix in range(new_data.shape[0]):
        state = new_data[row_ix, 0]
        if old_state != state:
            choices[state_start_row_ix + np.argmax(utilities[state_start_row_ix:row_ix])] = 1.0
            state_start_row_ix = row_ix
        old_state = state
    choices[state_start_row_ix + np.argmax(utilities[state_start_row_ix:])] = 1.0
    return choices


@njit
def stew_multinomial_logit_predicted_probabilities(new_data, weights):
    exp_utilities = np.exp(new_data[:, 1:].dot(weights))
    old_state = new_data[0, 0]
    state_start_row_ix = 0
    probabilities = np.zeros(len(new_data), np.float_)
    for row_ix in range(new_data.shape[0]):
        state = new_data[row_ix, 0]
        if old_state != state:
            probabilities[state_start_row_ix:row_ix] = exp_utilities[state_start_row_ix:row_ix] / np.sum(exp_utilities[state_start_row_ix:row_ix])
            state_start_row_ix = row_ix
        old_state = state
    probabilities[state_start_row_ix:] = exp_utilities[state_start_row_ix:] / np.sum(exp_utilities[state_start_row_ix:])
    return probabilities


@njit
def single_choice_set_grad(beta, data):
    grad = np.zeros(data.shape[1] - 2, dtype=np.float_)
    utilities = data[:, 2:].dot(beta)
    utilities = utilities-np.max(utilities)
    exp_utilities = np.exp(utilities)
    normalization_sum = np.sum(exp_utilities)
    grad += np.sum((data[:, 1] - exp_utilities / normalization_sum).reshape((-1, 1))
                   * data[:, 2:], axis=0)
    return grad


@njit
def stew_multinomial_logit_ll_and_grad(beta, data, D, lam):
    ll = 0.0
    grad = np.zeros(data.shape[1] - 2, dtype=np.float_)
    utilities = data[:, 2:].dot(beta)
    utilities = utilities-np.max(utilities)
    exp_utilities = np.exp(utilities)
    normalization_sum = 0.0
    old_state = data[0, 0]
    state_start_row_ix = 0
    for row_ix in range(data.shape[0]):
        state = data[row_ix, 0]
        if old_state != state:
            ll -= np.log(normalization_sum)
            # if normalization_sum > 1000000:
            #     print(normalization_sum)
            grad += np.sum((data[state_start_row_ix:row_ix, 1] - exp_utilities[state_start_row_ix:row_ix] / normalization_sum).reshape((-1, 1))
                           * data[state_start_row_ix:row_ix, 2:], axis=0)
            # Reset
            normalization_sum = 0.0
            state_start_row_ix = row_ix
        normalization_sum += exp_utilities[row_ix]
        choice = data[row_ix, 1]
        if choice == 1.0:
            ll += utilities[row_ix]
        old_state = state
    ll -= np.log(normalization_sum)
    grad += np.sum((data[state_start_row_ix:, 1] - exp_utilities[state_start_row_ix:] / normalization_sum).reshape((-1, 1))
                   * data[state_start_row_ix:, 2:], axis=0)
    if lam > 0:
        ll -= lam * beta.T.dot(D).dot(beta)
        grad -= 2 * lam * beta.dot(D)
    return -ll, -grad


