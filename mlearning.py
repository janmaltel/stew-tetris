import numpy as np
from domtools.domtools.domfilter import dom_filter
from tetris import tetromino
from stew.stew.mlogit import StewMultinomialLogit
from stew.stew.utils import create_diff_matrix, create_ridge_matrix


class MLearning:
    """
    M-learning, tailored to application in Tetris.
    """
    def __init__(self, feature_type,
                 num_columns,
                 verbose, verbose_stew,
                 lambda_min, lambda_max, num_lambdas,
                 dominance_filter, cumu_dom_filter, rollout_dom_filter, rollout_cumu_dom_filter,
                 gamma,
                 regularization,
                 rollout_length,
                 learn_every_step_until, avg_expands_per_children, feature_directors,
                 max_batch_size, learn_periodicity):
        self.feature_type = feature_type
        self.num_features = 8
        self.num_columns = num_columns  # ...of the Tetris board
        self.feature_names = np.array(['rows_with_holes', 'column_transitions', 'holes',
                                       'landing_height', 'cumulative_wells', 'row_transitions',
                                       'eroded', 'hole_depth'])  # Uses BCTS features.
        self.verbose = verbose
        self.verbose_stew = verbose_stew
        self.max_choice_set_size = 35  # There are never more than 34 actions in Tetris
        self.tetrominos = [tetromino.Straight(self.feature_type, self.num_features, self.num_columns),
                           tetromino.RCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.LCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.Square(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeR(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeL(self.feature_type, self.num_features, self.num_columns),
                           tetromino.T(self.feature_type, self.num_features, self.num_columns)]
        self.tetromino_sampler = tetromino.TetrominoSampler(self.tetrominos)
        self.learned_directions = np.zeros(self.num_features)
        self.feature_order = np.arange(self.num_features)
        self.step = 0
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.dom_filter = dominance_filter
        self.cumu_dom_filter = cumu_dom_filter
        self.rollout_dom_filter = rollout_dom_filter
        self.rollout_cumu_dom_filter = rollout_cumu_dom_filter
        self.delete_every = 2
        self.learn_from_step = 2
        self.avg_expands_per_children = avg_expands_per_children
        self.num_total_rollouts = self.rollout_length * self.avg_expands_per_children
        self.learn_periodicity = learn_periodicity
        self.learn_every_step_until = learn_every_step_until
        self.max_batch_size = max_batch_size
        if feature_directors is None:
            if self.feature_type == 'bcts':
                print("Features are directed automatically to be BCTS features.")
                self.feature_directors = np.array([-1, -1, -1, -1, -1, -1, 1, -1])
        else:
            self.feature_directors = feature_directors

        self.policy_weights = np.random.normal(loc=0.0, scale=0.1, size=self.num_features)

        # Data and model (regularization type)
        self.regularization = regularization
        assert(self.regularization in ["stew", "ols", "ridge", "nonnegative", "ew"])
        if self.regularization == "ridge":
            D = create_ridge_matrix(self.num_features)
        else:
            D = create_diff_matrix(self.num_features)
        self.model = StewMultinomialLogit(num_features=self.num_features, D=D, lambda_min=lambda_min,
                                          lambda_max=lambda_max, num_lambdas=num_lambdas, verbose=self.verbose_stew,
                                          nonnegative=self.regularization=="nonnegative")
        self.mlogit_data = MlogitData(num_features=self.num_features, max_choice_set_size=self.max_choice_set_size)
        self.rollout_tetrominos = None

    def reset_agent(self):
        self.step = 0

    def create_rollout_tetrominos(self):
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
            action_features[ix] = children_states[ix].get_features(direct_by=self.feature_directors, order_by=self.feature_order)
        if self.dom_filter or self.cumu_dom_filter:
            not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_children)  # domtools.
            if self.cumu_dom_filter:
                children_states = children_states[not_cumu_dominated]
                map_back_vector = np.nonzero(not_cumu_dominated)[0]
            else:  # Only simple dominance
                children_states = children_states[not_simply_dominated]
                map_back_vector = np.nonzero(not_simply_dominated)[0]
            num_children = len(children_states)
        else:
            map_back_vector = np.arange(num_children)
        child_total_values = np.zeros(num_children)
        self.create_rollout_tetrominos()
        for child in range(num_children):
            for rollout_ix in range(self.avg_expands_per_children):
                child_total_values[child] += self.roll_out(start_state=children_states[child], rollout_ix=rollout_ix)
        child_index = np.argmax(child_total_values)
        children_states[child_index].value_estimate = child_total_values[child_index]
        before_filter_index = map_back_vector[child_index]  # Needed for probabilities in gradient in learn()
        return children_states[child_index], before_filter_index, action_features

    def roll_out(self, start_state, rollout_ix):
        value_estimate = start_state.reward
        state_tmp = start_state
        count = 1
        while not state_tmp.terminal_state and count <= self.rollout_length:
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
                action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=self.feature_order)
            not_simply_dominated, not_cumu_dominated = dom_filter(action_features, len_after_states=num_states)  # domtools.
            if self.rollout_cumu_dom_filter:
                available_after_states = available_after_states[not_simply_dominated]
                action_features = action_features[not_simply_dominated]
            elif self.rollout_dom_filter:
                available_after_states = available_after_states[not_cumu_dominated]
                action_features = action_features[not_cumu_dominated]
            num_states = len(available_after_states)

        if action_features is None:  # Happens if no dom_filters have been applied.
            action_features = np.zeros((num_states, self.num_features))
            for ix, after_state in enumerate(available_after_states):
                action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=self.feature_order)
        utilities = action_features.dot(self.policy_weights)
        move_index = np.argmax(utilities)
        move = available_after_states[move_index]
        return move, move_index

    def learn(self, action_features, action_index):
        """
        Learns new policy weights from choice set data.
        """
        delete_oldest = self.mlogit_data.current_number_of_choice_sets > self.max_batch_size or (self.delete_every > 0 and self.step % self.delete_every == 0 and self.step >= self.learn_from_step + 1)
        self.mlogit_data.push(features=action_features, choice_index=action_index, delete_oldest=delete_oldest)
        if self.step >= self.learn_from_step and (self.step <= self.learn_every_step_until or self.step % self.learn_periodicity == self.learn_periodicity - 1):
            if self.regularization in ["ols", "nonnegative"]:
                self.policy_weights = self.model.fit(data=self.mlogit_data.sample(), lam=0, standardize=False)
            elif self.regularization in ["ridge", "stew"]:
                self.policy_weights, _ = self.model.cv_fit(data=self.mlogit_data.sample())

    def choose_action_test(self, start_state, start_tetromino):
        """
        Chooses actions in test mode, that is, without rollouts and without dominance filtering.
        Chooses the utility-maximising action.
        """
        all_available_after_states = start_tetromino.get_after_states(current_state=start_state)
        available_after_states = np.array([child for child in all_available_after_states if not child.terminal_state])
        if len(available_after_states) == 0:
            # Game over!
            return all_available_after_states[0], 0
        num_states = len(available_after_states)
        action_features = np.zeros((num_states, self.num_features))
        for ix, after_state in enumerate(available_after_states):
            action_features[ix] = after_state.get_features(direct_by=self.feature_directors, order_by=self.feature_order)
        utilities = action_features.dot(self.policy_weights)
        max_indices = np.where(utilities == np.max(utilities))[0]
        move_index = np.random.choice(max_indices)
        move = available_after_states[move_index]
        return move, move_index


class MlogitData(object):
    """
    Data structure to store choice sets. See Table 1 in the Supplementary Material of the article.
    """
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
        # Currently just returns entire data set.
        return self.data

    def delete_data(self):
        self.data = np.zeros((0, self.num_features + 2))

