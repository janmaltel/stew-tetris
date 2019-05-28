import numpy as np
import stew
from domtools.domtools.domfilter import dom_filter
from tetris import tetromino
from numba import njit
from scipy.stats import binom_test
from sklearn.linear_model import LinearRegression
from stew.stew.mlogit import StewMultinomialLogit
import gc
import cma
import time


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


class Cbmpi:
    def __init__(self, m, N, M, B, D, feature_type, num_columns, cmaes_var, verbose, seed=0, id="",
                 discrete_choice=False):
        self.num_features = 8
        self.num_value_features = 13
        self.num_columns = num_columns
        self.feature_type = feature_type
        self.verbose = verbose
        self.max_choice_set_size = 35
        self.tetrominos = [tetromino.Straight(self.feature_type, self.num_features, self.num_columns),
                           tetromino.RCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.LCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.Square(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeR(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeL(self.feature_type, self.num_features, self.num_columns),
                           tetromino.T(self.feature_type, self.num_features, self.num_columns)]
        self.tetromino_sampler = tetromino.TetrominoSampler(self.tetrominos)

        self.m = m  # rollout_size
        self.M = M  # number of independent rollouts per
        assert (self.M == 1)
        self.B = B  # budget
        if N is None:
            self.N = int(self.B / self.M / (self.m+1) / 32)
            print("Given the budget of", self.B, " the rollout set size will be:", self.N)
        else:
            self.N = N  # number of states sampled from rollout_set D_k
        self.D = D  # rollout set
        self.D_k = None
        # # TODO: put in learn??
        # self.construct_rollout_set()

        # print("The budget is", self.B)
        self.gamma = 1
        self.eta = 0  # CMA-ES parameter (dunno where to set...)
        self.zeta = 0.5  # CMA-ES parameter ( // 2 is standard in cma)
        self.n = int(self.num_features * 15)  # CMA-ES parameter ('popsize' in cma)
        self.cmaes_var = cmaes_var

        self.policy_weights = np.random.normal(loc=0, scale=self.cmaes_var, size=self.num_features)
        self.value_weights = np.zeros(self.num_value_features)

        self.lin_reg = LinearRegression(n_jobs=1, fit_intercept=True)  # n_jobs must be 1... otherwise clashes with multiprocessing.Pool
        self.lin_reg.coef_ = self.value_weights
        self.lin_reg.intercept_ = 0.

        self.seed = seed

        self.discrete_choice = discrete_choice
        if self.discrete_choice:
            self.ols = True
            self.model = stew.StewMultinomialLogit(num_features=self.num_features)
            self.mlogit_data = MlogitData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)
        else:
            self.cma_es = cma.CMAEvolutionStrategy(np.random.normal(loc=0, scale=1, size=self.num_features), self.cmaes_var, inopts={'verb_disp': 0,
                                                                                   'verb_filenameprefix': "cmaesout" + str(id) + "_" + str(self.seed),
                                                                                   'popsize': self.n})

    def learn(self):
        self.construct_rollout_set()
        if self.verbose:
            print("Rollout set constructed")
        start_time = time.time()
        self.state_features = np.zeros((self.N, self.num_value_features), dtype=np.float)
        self.state_values = np.zeros(self.N, dtype=np.float)
        self.state_action_values = np.zeros((self.N, 34), dtype=np.float)
        # self.state_action_features = []
        self.state_action_features = np.zeros((self.N, 34, self.num_features))
        self.num_available_actions = np.zeros(self.N, dtype=int)
        self.did_rollout = np.ones(self.N, dtype=bool)
        for ix, rollout_state in enumerate(self.D_k):
            # if ix % 5000 == 0 and self.verbose:
            #     print("rollout state:", ix)
            self.state_features[ix, :] = rollout_state.get_features(direct_by=None, order_by=None, standardize_by=None, addRBF=True)
            self.state_values[ix] = self.value_roll_out(rollout_state)
            actions_value_estimates, state_action_features = self.action_value_roll_out(rollout_state)
            num_available_actions = len(actions_value_estimates)
            self.num_available_actions[ix] = num_available_actions
            self.state_action_values[ix, :num_available_actions] = actions_value_estimates
            if state_action_features is not None:
                self.state_action_features[ix, :num_available_actions, :] = state_action_features
            else:
                self.did_rollout[ix] = False
            # self.state_action_features.append(state_action_features)

        self.delete_rollout_set()
        end_time = time.time()
        if self.verbose:
            print("Rollouts took " + str((end_time - start_time) / 60) + " minutes.")
        assert(len(self.state_action_features) == self.N)
        # # Max
        # self.max_state_action_values = self.state_action_values.max(axis=1)

        if self.verbose:
            print("Estimating weights now...")

        # Approximate value function
        if self.verbose:
            print("Number of self.state_values", len(self.state_values))
        self.lin_reg.fit(self.state_features, self.state_values)
        self.value_weights = np.hstack((self.lin_reg.intercept_, self.lin_reg.coef_))
        if self.verbose:
            print("new value weights: ", self.value_weights)

        # Approximate policy
        start_time = time.time()
        if self.discrete_choice:
            self.mlogit_data.delete_data()
            # stacked_features = np.concatenate([self.state_action_features, axis=0)
            # TODO: rewrite to use less copying / memory...
            for state_ix in range(self.N):
                state_act_feat_ix = self.state_action_features[state_ix]
                if state_act_feat_ix is not None:
                    action_values = self.state_action_values[state_ix, :len(state_act_feat_ix)]
                    self.mlogit_data.push(features=self.state_action_features[state_ix], choice_index=np.argmax(action_values), delete_oldest=False)
            if self.ols:
                self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
            # else:
            #     self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample(), standardize=self.standardize_features)
        else:
            self.cma_es = cma.CMAEvolutionStrategy(np.random.normal(loc=0, scale=1, size=self.num_features), self.cmaes_var,
                                                   inopts={'verb_disp': 0,
                                                           'verb_filenameprefix': "cmaesout" + str(id) + "_" + str(self.seed),
                                                           'popsize': self.n})
            self.policy_weights = self.cma_es.optimize(self.policy_loss_function, min_iterations=1).result.xbest
        end_time = time.time()
        if self.verbose:
            print("CMAES took " + str((end_time - start_time) / 60) + " minutes.")
            print("new policy_weights: ", self.policy_weights)

    def construct_rollout_set(self):
        self.D_k = np.random.choice(a=self.D, size=self.N, replace=False)

    def delete_rollout_set(self):
        self.D_k = None
        gc.collect()

    def policy_loss_function(self, pol_weights):
        loss = 0.
        number_of_samples = 0
        for state_ix in range(self.N):
            # print("self.N", self.N)
            if self.did_rollout[state_ix]:
                values = self.state_action_features[state_ix, :self.num_available_actions[state_ix]].dot(pol_weights)
                # print("values", values)
                pol_value = self.state_action_values[state_ix, np.argmax(values)]
                max_value = np.max(self.state_action_values[state_ix, :len(values)])
                # print("state_ix", state_ix)
                # print("self.state_action_features[state_ix]", self.state_action_features[state_ix])
                # print("pol_weights", pol_weights)
                loss += max_value - pol_value
                number_of_samples += 1
            # else:
            #     pass
            #     # print(state_ix, " has no action features / did not produce a rollout!")
        loss /= number_of_samples
        # print("loss", loss)
        return loss

    def value_roll_out(self, start_state):
        # value_estimate = start_state.reward
        value_estimate = 0.0
        state_tmp = start_state
        count = 0
        game_ended = False
        # print("------------------------------------")
        # print("------------------------------------")
        # print("Starting new value rollout")
        # print("------------------------------------")
        # print("------------------------------------")
        while not game_ended and count < self.m:  # there are only (m-1) rollouts
            tetromino_tmp = self.tetromino_sampler.next_tetromino()
            # print("state_tmp.representation")
            # print(state_tmp.representation)
            state_tmp, _ = self.choose_action_value_rollout(start_state=state_tmp, start_tetromino=tetromino_tmp)
            if state_tmp is None:
                game_ended = True
            else:
                value_estimate += (self.gamma ** count) * state_tmp.reward
            count += 1

        # One more (the m-th) for truncation value!
        if not game_ended:
            tetromino_tmp = self.tetromino_sampler.next_tetromino()
            state_tmp, _ = self.choose_action_value_rollout(start_state=state_tmp, start_tetromino=tetromino_tmp)
            if state_tmp is not None:
                final_state_features = state_tmp.get_features(direct_by=None, order_by=None, standardize_by=None, addRBF=True)
                # value_estimate += self.gamma ** count + final_state_features.dot(self.value_weights)
                value_estimate += (self.gamma ** count) * self.lin_reg.predict(final_state_features.reshape(1, -1))
        return value_estimate

    def action_value_roll_out(self, start_state):
        start_tetromino = self.tetromino_sampler.next_tetromino()
        all_children_states = start_tetromino.get_after_states(current_state=start_state)
        children_states = np.array([child for child in all_children_states if not child.terminal_state])
        num_children = len(children_states)
        action_value_estimates = np.zeros(num_children)
        state_action_features = np.zeros((num_children, self.num_features))
        if num_children == 0:
            # TODO: check whether this (returning zeros) has any side effects on learning...
            # Use the None (instead of state_action_features) to propagate the knowledge that there was no decision to make here.
            return action_value_estimates, None
        for child_ix in range(num_children):
            state_tmp = children_states[child_ix]
            state_action_features[child_ix] = state_tmp.get_features(direct_by=None, order_by=None, standardize_by=None)
            cumulative_reward = state_tmp.reward
            # cumulative_reward = 0

            # print("------------------------------------")
            # print("------------------------------------")
            # print("Starting new action rollout")
            # print("------------------------------------")
            # print("------------------------------------")
            game_ended = False
            count = 0
            while not game_ended and count < self.m:  # there are m rollouts
                tetromino_tmp = self.tetromino_sampler.next_tetromino()
                # print("state_tmp.representation")
                # print(state_tmp.representation)
                state_tmp, _ = self.choose_action_q_rollout(start_state=state_tmp, start_tetromino=tetromino_tmp)
                if state_tmp is None:
                    game_ended = True
                else:
                    cumulative_reward += (self.gamma ** count) * state_tmp.reward
                count += 1

            # One more (the (m+1)-th) for truncation value!
            if not game_ended:
                tetromino_tmp = self.tetromino_sampler.next_tetromino()
                state_tmp, _ = self.choose_action_q_rollout(start_state=state_tmp, start_tetromino=tetromino_tmp)
                if state_tmp is not None:
                    final_state_features = state_tmp.get_features(direct_by=None, order_by=None, standardize_by=None, addRBF=True)
                    # cumulative_reward += self.gamma ** count + final_state_features.dot(self.value_weights)
                    cumulative_reward += (self.gamma ** count) * self.lin_reg.predict(final_state_features.reshape(1, -1))

            action_value_estimates[child_ix] = cumulative_reward
        return action_value_estimates, state_action_features

    def choose_action_value_rollout(self, start_state, start_tetromino):
        return self.choose_action_q_rollout(start_state, start_tetromino)

    def choose_action_q_rollout(self, start_state, start_tetromino):
        # Returning "None" if game is over.
        all_available_after_states = start_tetromino.get_after_states(current_state=start_state)
        available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        num_states = len(available_after_states)
        if num_states == 0:
            # Game over!
            return None, 0
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=None, order_by=None, standardize_by=None)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index

    def choose_action_test(self, start_state, start_tetromino):
        # Returning the first state if game is over.
        all_available_after_states = start_tetromino.get_after_states(current_state=start_state)
        available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        num_states = len(available_after_states)
        if num_states == 0:
            # Game over!
            return all_available_after_states[0], 0
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=None, order_by=None, standardize_by=None)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index

    def choose_action_test_hard(self, start_state, start_tetromino):
        # Returning the first state if game is over.
        available_after_states = start_tetromino.get_after_states(current_state=start_state)
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=None, order_by=None, standardize_by=None)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index


# @njit
# def policy_loss_function_jit(pol_weights, N, state_action_features, state_action_values):
#     loss = 0.
#     number_of_samples = 0
#     for state_ix in range(N):
#         if state_action_features[state_ix] is not None:
#             values = state_action_features[state_ix].dot(pol_weights)
#             # print("values", values)
#             pol_value = state_action_values[state_ix, np.argmax(values)]
#             max_value = np.max(state_action_values[state_ix, :len(values)])
#             # print("state_ix", state_ix)
#             # print("self.state_action_features[state_ix]", self.state_action_features[state_ix])
#             # print("pol_weights", pol_weights)
#             loss += max_value - pol_value
#             number_of_samples += 1
#         else:
#             pass
#             # print(state_ix, " has no action features / did not produce a rollout!")
#     loss /= number_of_samples
#     # print("loss", loss)
#     return loss


class HierarchicalLearner:
    def __init__(self, start_phase_ix, feature_type, num_columns,
                 verbose, verbose_stew, lambda_min, lambda_max, num_lambdas,
                 dom_filter_phases, cumu_dom_filter_phases, rollout_dom_filter_phases,
                 rollout_cumu_dom_filter_phases, filter_best_phases, ols_phases,
                 max_length_phases, gamma_phases, rollout_length_phases, rollout_action_selection_phases,
                 delete_every_phases, learn_from_step_phases, learn_every_step_until, avg_expands_per_children_phases, feature_directors,
                 standardize_features, max_batch_size, learn_every_after, ew=False,
                 random_init_weights=False, do_sgd_update=False, ridge=False, one_se_rule=False,
                 nonnegative=False):
        self.do_sgd_update = do_sgd_update
        self.feature_type = feature_type
        self.num_features = 8
        self.standardize_features = standardize_features
        if self.feature_type == "bcts":
            self.feature_names = np.array(['rows_with_holes', 'column_transitions', 'holes',
                                           'landing_height', 'cumulative_wells', 'row_transitions',
                                           'eroded', 'hole_depth'])
        elif self.feature_type == "standardized_bcts":
            self.feature_names = np.array(['eroded', 'rows_with_holes', 'landing_height',
                                           'column_transitions', 'holes', 'cumulative_wells',
                                           'row_transitions', 'hole_depth'])
        print("Started with", self.feature_names)
        self.num_columns = num_columns
        self.verbose = verbose
        self.verbose_stew = verbose_stew
        self.max_choice_set_size = 35
        self.tetrominos = [tetromino.Straight(self.feature_type, self.num_features, self.num_columns),
                           tetromino.RCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.LCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.Square(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeR(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeL(self.feature_type, self.num_features, self.num_columns),
                           tetromino.T(self.feature_type, self.num_features, self.num_columns)]
        self.tetromino_sampler = tetromino.TetrominoSampler(self.tetrominos)
        self.positive_direction_counts = np.zeros(self.num_features)
        self.meaningful_comparisons = np.zeros(self.num_features)
        self.learned_directions = np.zeros(self.num_features)
        self.feature_order = np.arange(self.num_features)
        self.learned_order = np.arange(self.num_features)
        self.feature_stds = np.ones(self.num_features)
        self.step = 0
        self.step_in_new_phase = 0
        self.gamma_phases = gamma_phases
        self.rollout_length_phases = rollout_length_phases
        self.rollout_action_selection_phases = rollout_action_selection_phases
        self.max_length_phases = max_length_phases
        self.dom_filter_phases = dom_filter_phases
        self.cumu_dom_filter_phases = cumu_dom_filter_phases
        self.rollout_dom_filter_phases = rollout_dom_filter_phases
        self.rollout_cumu_dom_filter_phases = rollout_cumu_dom_filter_phases
        self.filter_best_phases = filter_best_phases
        self.ols_phases = ols_phases
        self.delete_every_phases = delete_every_phases
        self.learn_from_step_phases = learn_from_step_phases
        self.avg_expands_per_children_phases = avg_expands_per_children_phases

        # Phase init
        self.phase_names = ["learn_directions", "learn_order", "learn_weights", "optimize_weights"]
        self.num_phases = len(self.phase_names)
        self.phase_ix = start_phase_ix
        self.phase = self.phase_names[self.phase_ix]
        if self.verbose:
            print("Starting with phase: ", self.phase)

        self.gamma = self.gamma_phases[self.phase_ix]
        self.rollout_length = self.rollout_length_phases[self.phase_ix]
        self.rollout_action_selection = self.rollout_action_selection_phases[self.phase_ix]
        self.max_length_phase = self.max_length_phases[self.phase_ix]
        self.dom_filter = self.dom_filter_phases[self.phase_ix]
        self.cumu_dom_filter = self.cumu_dom_filter_phases[self.phase_ix]
        self.rollout_dom_filter = self.rollout_dom_filter_phases[self.phase_ix]
        self.rollout_cumu_dom_filter = self.rollout_cumu_dom_filter_phases[self.phase_ix]
        self.filter_best = self.filter_best_phases[self.phase_ix]
        self.ols = self.ols_phases[self.phase_ix]
        self.delete_every = self.delete_every_phases[self.phase_ix]
        self.learn_from_step = self.learn_from_step_phases[self.phase_ix]
        self.avg_expands_per_children = self.avg_expands_per_children_phases[self.phase_ix]
        self.num_total_rollouts = self.rollout_length * self.avg_expands_per_children
        self.learn_every_after = learn_every_after
        self.learn_every_step_until = learn_every_step_until

        self.ew = ew
        self.max_batch_size = max_batch_size
        # self.switch_to_ol = switch_to_ol
        if feature_directors is None:
            if self.feature_type == 'bcts' and self.phase == "learn_weights":
                print("Features are directed automatically.")
                self.feature_directors = np.array([-1, -1, -1, -1, -1, -1, 1, -1])
            else:
                self.feature_directors = np.ones(self.num_features)
        else:
            self.feature_directors = feature_directors

        # TODO: change back to ones?
        # self.policy_weights = np.ones(self.num_features) / 10
        if random_init_weights:
            self.policy_weights = np.random.normal(loc=0.0, scale=0.1, size=self.num_features)
        else:
            self.policy_weights = np.ones(self.num_features)
        # self.policy_weights = np.zeros(self.num_features) / 10

        # Data and model
        self.ridge = ridge
        if self.ridge:
            D = create_ridge_matrix(self.num_features)
        else:
            D = create_diff_matrix(self.num_features)

        self.nonnegative = nonnegative
        # stew.S....
        self.model = StewMultinomialLogit(num_features=self.num_features, D=D, lambda_min=lambda_min,
                                               lambda_max=lambda_max, num_lambdas=num_lambdas, verbose=self.verbose_stew,
                                               one_se_rule=one_se_rule, nonnegative=nonnegative)
        self.mlogit_data = MlogitData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)
        self.rollout_tetrominos = None
        self.gen_model_count = 0

    def reset_agent(self):
        self.step = 0
        self.step_in_new_phase = 0

    def create_rollout_tetrominos(self):
        # TODO: create functionality that makes sure that every tetromino is used at least once in the first row.
        self.rollout_tetrominos = np.array([self.tetromino_sampler.next_tetromino() for _ in range(self.num_total_rollouts)])
        self.rollout_tetrominos.shape = (self.rollout_length, self.avg_expands_per_children)

    def choose_action(self, start_state, start_tetromino):
        all_children_states = start_tetromino.get_after_states(current_state=start_state)
        children_states = np.array([child for child in all_children_states if not child.terminal_state])
        if len(children_states) == 0:
            # Game over!
            return all_children_states[0], 0, None
        num_children = len(children_states)
        action_features = np.zeros((num_children, self.num_features), dtype=np.float_)
        for ix in range(num_children):
            self.gen_model_count += 1
            action_features[ix] = children_states[ix].get_features(direct_by=self.feature_directors, order_by=self.feature_order, standardize_by=self.feature_stds)
        if self.dom_filter or self.cumu_dom_filter:
            not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_children)  # domtools.
            if self.cumu_dom_filter:
                children_states = children_states[not_cumu_dominated]
                map_back_vector = np.nonzero(not_cumu_dominated)[0]
            else:  # Only simple dom
                children_states = children_states[not_simply_dominated]
                map_back_vector = np.nonzero(not_simply_dominated)[0]
            num_children = len(children_states)
        elif self.filter_best:
            utilities = action_features.dot(self.policy_weights)
            map_back_vector = np.argsort(-utilities)[:5]
            children_states = children_states[map_back_vector]
        else:
            map_back_vector = np.arange(num_children)
        child_total_values = np.zeros(num_children)
        self.create_rollout_tetrominos()
        for child in range(num_children):
            for rollout_ix in range(self.avg_expands_per_children):
                child_total_values[child] += self.roll_out(start_state=children_states[child], rollout_ix=rollout_ix)
        child_index = np.argmax(child_total_values)
        # if self.verbose:
        #     print("child_total_values")
        #     print(child_total_values)
        children_states[child_index].value_estimate = child_total_values[child_index]
        before_filter_index = map_back_vector[child_index]  # Needed for probabilities in gradient in learn()
        return children_states[child_index], before_filter_index, action_features

    def roll_out(self, start_state, rollout_ix):
        value_estimate = start_state.reward
        state_tmp = start_state
        count = 1
        while not state_tmp.terminal_state and count <= self.rollout_length:
            # tetromino_tmp = self.tetromino_sampler.next_tetromino()
            tetromino_tmp = self.rollout_tetrominos[count-1, rollout_ix]
            all_available_after_states = tetromino_tmp.get_after_states(state_tmp)
            non_terminal_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
            if len(non_terminal_after_states) == 0:
                # Game over!
                return value_estimate
            state_tmp, _ = self.choose_rollout_action(non_terminal_after_states)
            value_estimate += self.gamma ** count * state_tmp.reward
            count += 1
        return value_estimate

    def choose_rollout_action(self, available_after_states):
        num_states = len(available_after_states)
        action_features = None
        if self.rollout_dom_filter or self.rollout_cumu_dom_filter:
            action_features = np.zeros((num_states, self.num_features))
            for ix, after_state in enumerate(available_after_states):
                self.gen_model_count += 1
                action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=self.feature_order, standardize_by=self.feature_stds)
            not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_states)  # domtools.
            if self.rollout_cumu_dom_filter:
                available_after_states = available_after_states[not_simply_dominated]
                action_features = action_features[not_simply_dominated]
            elif self.rollout_dom_filter:
                available_after_states = available_after_states[not_cumu_dominated]
                action_features = action_features[not_cumu_dominated]
            num_states = len(available_after_states)
        if self.rollout_action_selection == "random":
            rewards = np.zeros(num_states)
            for ix, after_state in enumerate(available_after_states):
                rewards[ix] = after_state.n_cleared_lines
            if np.any(rewards):
                move_index = int(np.random.choice(a=np.where(rewards)[0], size=1))
            else:
                move_index = int(np.random.choice(a=np.arange(num_states), size=1))
        elif self.rollout_action_selection == "max_util":
            if action_features is None:  # Happens if no dom_filters have been applied.
                action_features = np.zeros((num_states, self.num_features))
                for ix, after_state in enumerate(available_after_states):
                    action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=self.feature_order, standardize_by=self.feature_stds)
            utilities = action_features.dot(self.policy_weights)
            move_index = np.argmax(utilities)
        move = available_after_states[move_index]
        return move, move_index

    def learn(self, action_features, action_index):
        if self.phase in ["learn_directions", "learn_order"]:  # self.phase == "learn_directions"
            chosen_action_features = action_features[action_index]
            remaining_action_features = np.delete(arr=action_features, obj=action_index, axis=0)
            feature_differences = np.sign(chosen_action_features - remaining_action_features)
            direction_counts = np.sign(np.sum(feature_differences, axis=0))
            self.positive_direction_counts += np.maximum(direction_counts, 0)
            self.meaningful_comparisons += np.abs(direction_counts)
            switched_phase = self.check_phase()
            self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=False)
        elif self.phase in ["learn_weights", "optimize_weights"]:
            if self.do_sgd_update:
                choice_set_len = len(action_features)
                one_hot_choice = np.zeros((choice_set_len, 1))
                one_hot_choice[action_index] = 1.
                choice_set_index = np.full(shape=(choice_set_len, 1), fill_value=1)
                data = np.hstack((choice_set_index, one_hot_choice, action_features))
                self.policy_weights = self.model.sgd_update(weights=self.policy_weights, data=data)
            else:
                delete_oldest = self.mlogit_data.current_number_of_choice_sets > self.max_batch_size or (self.delete_every > 0 and self.step_in_new_phase % self.delete_every == 0 and self.step_in_new_phase >= self.learn_from_step + 1)
                self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=delete_oldest)
                if self.step_in_new_phase >= self.learn_from_step and (self.step_in_new_phase <= self.learn_every_step_until or self.step_in_new_phase % self.learn_every_after == self.learn_every_after-1):
                    if self.ols or self.nonnegative:
                        self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
                    else:
                        self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample(), standardize=self.standardize_features)
            switched_phase = self.check_phase()
        return switched_phase

    def check_phase(self):
        switched_phase = False
        self.step_in_new_phase += 1
        if self.phase == "learn_directions":
            if self.step_in_new_phase >= 5:
                unidentified_directions = np.where(self.learned_directions == 0.)[0]
                for feature_ix in range(len(unidentified_directions)):
                    feature = unidentified_directions[feature_ix]
                    p_value = binom_test(x=self.positive_direction_counts[feature],
                                         n=self.meaningful_comparisons[feature],
                                         p=0.5, alternative="two-sided")
                    if self.verbose:
                        print("Feature ", feature, " has ", self.positive_direction_counts[feature], "/", self.meaningful_comparisons[feature],
                              "pos. comp. P-value: ", np.round(p_value, 4))
                    if p_value < 0.05:
                        self.learned_directions[feature] = np.sign(self.positive_direction_counts[feature] / self.meaningful_comparisons[feature] - 0.5)
                        if self.verbose:
                            print("Feature ", feature, " has been decided to be: ", self.learned_directions[feature])
                if np.all(self.learned_directions != 0.):
                    if self.verbose:
                        print("All directions have been identified. They are:")
                        print(self.learned_directions)
                    self.switch_phase()
                    switched_phase = True
        elif self.phase == "learn_order":
            temp_ratios = self.positive_direction_counts / self.meaningful_comparisons
            self.learned_order = np.argsort(-temp_ratios)
            sorted_ratios = temp_ratios[self.learned_order]
            p_values = np.zeros(self.num_features-1)
            for ix in range(len(sorted_ratios)-1):
                stat, p_value = proportions_ztest(self.positive_direction_counts[ix:ix+2], self.meaningful_comparisons[ix:ix+2])
                if self.verbose:
                    print("Features ", ix, " and ", ix + 1, " have proportions of ", sorted_ratios[ix], " and ", sorted_ratios[ix+1])
                    print("The proportion_z_test shows a p-value of: ", np.round(p_value, 4))
                p_values[ix] = p_value
            if np.all(p_values < 0.1):
                print("The complete order has been identified. It is:")
                print(self.learned_order)
                print("The p-values are:")
                print(p_values)
                print("Switching to 'learn_weights' phase.")
                self.switch_phase()
                switched_phase = True
        elif self.phase == "learn_weights":
            pass
        else:
            raise ValueError("self.phase wrongly specified!")
        if self.step_in_new_phase > self.max_length_phase:
            print("SWITCHING PHASE because max_length_phase of " + str(self.max_length_phase) + " has been reached!")
            self.switch_phase()
            switched_phase = True
        return switched_phase

    # def push_data(self, action_features, action_index):
    #     self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=False)

    def switch_phase(self):
        self.step_in_new_phase = 0
        self.phase_ix += 1
        self.phase = self.phase_names[self.phase_ix]
        if self.verbose:
            print("Switching to phase: ", self.phase)

        self.gamma = self.gamma_phases[self.phase_ix]
        self.rollout_length = self.rollout_length_phases[self.phase_ix]
        self.rollout_action_selection = self.rollout_action_selection_phases[self.phase_ix]
        self.max_length_phase = self.max_length_phases[self.phase_ix]
        self.dom_filter = self.dom_filter_phases[self.phase_ix]
        self.cumu_dom_filter = self.cumu_dom_filter_phases[self.phase_ix]
        self.rollout_dom_filter = self.rollout_dom_filter_phases[self.phase_ix]
        self.rollout_cumu_dom_filter = self.rollout_cumu_dom_filter_phases[self.phase_ix]
        self.filter_best = self.filter_best_phases[self.phase_ix]
        self.ols = self.ols_phases[self.phase_ix]
        self.delete_every = self.delete_every_phases[self.phase_ix]
        self.learn_from_step = self.learn_from_step_phases[self.phase_ix]
        self.avg_expands_per_children = self.avg_expands_per_children_phases[self.phase_ix]
        self.num_total_rollouts = self.rollout_length * self.avg_expands_per_children

        if self.phase == "learn_order":  # coming from "learn_directions"
            print("The learned directions are: ", self.learned_directions)
            self.feature_directors = np.sign(self.positive_direction_counts / self.meaningful_comparisons - 0.5)
            print("Feature directors will be: ", self.feature_directors)
            # self.feature_directors = self.learned_directions
            self.positive_direction_counts = flip_positive_direction_counts(self.positive_direction_counts,
                                                                            self.meaningful_comparisons,
                                                                            self.feature_directors)

        elif self.phase == "learn_weights":  # coming from "learn_order"
            # Feature variance business
            print("Estimating feature variance from", len(self.mlogit_data.data), "data points")
            std_deviations = np.std(self.mlogit_data.data[:, 2:], axis=0)
            self.feature_stds = std_deviations
            print("The standard deviations are ", self.feature_stds)
            print("... for features", self.feature_names)
            self.mlogit_data.delete_data()

            # Order
            print("The original order was", self.feature_names)
            self.feature_order = self.feature_order[self.learned_order]
            self.feature_names = self.feature_names[self.learned_order]
            self.feature_stds = self.feature_stds[self.learned_order]
            print("The new order is", self.feature_names)
            self.feature_directors = self.feature_directors[self.learned_order]
            print("...and accordingly, the new directions are: ", self.feature_directors)

    def choose_action_test(self, start_state, start_tetromino):
        all_available_after_states = start_tetromino.get_after_states(current_state=start_state)
        available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        if len(available_after_states) == 0:
            # Game over!
            return all_available_after_states[0], 0
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=self.feature_order, standardize_by=self.feature_stds)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index


@njit
def flip_positive_direction_counts(positive_direction_counts, meaningful_comparisons, feature_directors):
    for ix in range(len(positive_direction_counts)):
        if feature_directors[ix] == -1.:
            positive_direction_counts[ix] = meaningful_comparisons[ix] - positive_direction_counts[ix]
    return positive_direction_counts


class MlogitData(object):
    def __init__(self, num_features, max_choice_set_size):
        self.num_features = num_features
        self.data = np.zeros((0, self.num_features + 2))
        self.choice_set_counter = 0.
        self.current_number_of_choice_sets = 0.
        self.max_choice_set_size = max_choice_set_size

    def push(self, features, choice_index, delete_oldest=False):
        choice_set_len = len(features)
        one_hot_choice = np.zeros((choice_set_len, 1))
        one_hot_choice[choice_index] = 1.
        choice_set_index = np.full(shape=(choice_set_len, 1), fill_value=self.choice_set_counter)
        self.data = np.vstack((self.data, np.hstack((choice_set_index, one_hot_choice, features))))
        self.choice_set_counter += 1.
        self.current_number_of_choice_sets += 1.
        if delete_oldest:
            first_choice_set_index = self.data[0, 0]
            for ix in range(self.max_choice_set_size):
                if self.data[ix, 0] != first_choice_set_index:
                    break
            if ix > self.max_choice_set_size:
                raise ValueError("Choice set should not be higher than 34.")
            self.data = self.data[ix:]
            if self.current_number_of_choice_sets > 0:
                self.current_number_of_choice_sets -= 1.


    def sample(self):
        # TODO: add sample_size (in terms of choice sets!)
        return self.data

    def delete_data(self):
        self.data = np.zeros((0, self.num_features + 2))


# def p_loop(p, seed, plot_individual=False):
#     random.seed(seed * p.seed)
#     np.random.seed(seed * p.seed)
#     # torch.manual_seed(seed)
#
#     player = HierarchicalLearner(start_phase_ix=p.start_phase_ix, feature_type=p.feature_type,
#                                  num_columns=p.num_columns,
#                                  verbose=p.verbose, verbose_stew=p.verbose_stew,
#                                  learn_from_step_phases=p.learn_from_step_phases,
#                                  avg_expands_per_children_phases=p.avg_expands_per_children_phases,
#                                  delete_every_phases=p.delete_every_phases,
#                                  lambda_min=p.lambda_min, lambda_max=p.lambda_max,
#                                  num_lambdas=p.num_lambdas,
#                                  gamma_phases=p.gamma_phases,
#                                  rollout_length_phases=p.rollout_length_phases,
#                                  rollout_action_selection_phases=p.rollout_action_selection_phases,
#                                  max_length_phases=p.max_length_phases,
#                                  dom_filter_phases=p.dom_filter_phases,
#                                  cumu_dom_filter_phases=p.cumu_dom_filter_phases,
#                                  rollout_dom_filter_phases=p.rollout_dom_filter_phases,
#                                  rollout_cumu_dom_filter_phases=p.rollout_cumu_dom_filter_phases,
#                                  filter_best_phases=p.filter_best_phases,
#                                  ols_phases=p.ols_phases,
#                                  feature_directors=p.feature_directors,
#                                  standardize_features=p.standardize_features)
#
#     environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=p.verbose)
#     test_environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=False,
#                                    max_cleared_test_lines=p.max_cleared_test_lines)
#
#     start = time.time()
#     testing_time = 0
#     _, test_results_ix, _2, tested_weights_ix, weights_storage_ix = \
#         environment.play_hierarchical_learning(plots_path=p.plots_path,
#                                                plot_analysis_fc=plot_analysis,
#                                                test_every=p.test_every,
#                                                num_tests=p.num_tests, num_test_games=p.num_test_games,
#                                                test_environment=test_environment,
#                                                testing_time=testing_time,
#                                                agent_ix=seed,
#                                                store_features=True)
#     end = time.time()
#     total_time = end - start
#
#     if plot_individual:
#         plots_path_ind = os.path.join(p.plots_path, "agent" + str(seed))
#         if not os.path.exists(plots_path_ind):
#             os.makedirs(plots_path_ind, exist_ok=True)
#         plot_individual_agent(plots_path=plots_path_ind, tested_weights=tested_weights_ix, test_results=test_results_ix,
#                               weights_storage=weights_storage_ix, agent_ix=seed)
#
#     return [test_results_ix, tested_weights_ix, weights_storage_ix, total_time]


