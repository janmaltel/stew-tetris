import numpy as np
from tetris import state, tetromino
import pprint
import collections
import time
import copy
# feature_names = ["holes", "cumulative_wells", "cumulative_wells_squared",
#                  "landing_height", "avg_free_row", "avg_free_row_squared", "n_landing_positions"]

# feature_names = ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
#                  'row_transitions', 'eroded', 'hole_depth']


class ToyState:
    def __init__(self, state_id, mdp):
        self.state_id = state_id
        self.mdp = mdp

    def act(self, action):
        if np.random.random() < self.mdp.term_eps:
            # Go to terminal state
            new_state_id = self.mdp.num_nonterminal_states
            game_over = True
        else:
            game_over = False
            if action == 0:  # clockwise
                if self.state_id < self.mdp.num_nonterminal_states - 1:
                    new_state_id = self.state_id + 1
                else:
                    new_state_id = 0
            elif action == 1:
                if self.state_id > 0:
                    new_state_id = self.state_id - 1
                else:
                    new_state_id = self.mdp.num_nonterminal_states - 1

            else:
                raise ValueError("Weird action choice...")
        reward = self.mdp.rewards[new_state_id]
        new_state = ToyState(int(new_state_id), mdp=self.mdp)
        return new_state, reward, game_over

    def get_features(self):
        features = np.zeros(self.mdp.num_nonterminal_states + 1)
        features[self.state_id] = 1
        # Only use p-1 features.
        return features[1:]


class ToyMDP:
    """
    States
             1
            / \
          0----2
                       Term

    Action 0 is going clockwise. 1 is anti-clockwise

    """
    def __init__(self, player, num_nonterminal_states=3, term_eps=0.001, terminal_reward=-100, verbose=False):
        self.num_nonterminal_states = num_nonterminal_states
        self.term_eps = term_eps
        self.nonterminal_states = np.arange(self.num_nonterminal_states)
        self.rewards = np.hstack((np.arange(self.num_nonterminal_states) + 1, np.array(terminal_reward)))
        self.player = player
        self.verbose = verbose

        self.current_state = new_state = ToyState(np.random.choice(self.num_nonterminal_states), self)
        self.actions = []
        self.action_features = []
        self.states = [new_state]
        self.ep_rewards = []
        self.game_over = False
        self.cumulative_steps = 0

    def reset(self):
        self.game_over = False
        self.current_state = new_state = ToyState(np.random.choice(self.num_nonterminal_states), self)
        self.actions = []
        self.action_features = []
        self.states = [new_state]
        self.ep_rewards = []

    def test_agent(self, n_episodes):
        rewards = np.zeros(n_episodes)
        for episode in range(n_episodes):
            self.reset()
            cumulative_reward = 0
            while not self.game_over:
                available_actions = self.get_available_actions()
                chosen_action = self.player.choose_action_test(available_actions, env=self)
                new_state, reward, game_over = self.act(chosen_action)
                self.current_state = new_state
                self.game_over = game_over
                cumulative_reward += reward
            rewards[episode] = cumulative_reward
        return np.mean(rewards)

    def play_and_test_R(self, num_tests=1, num_test_games=0, test_every=0, test_environment=None,
                        save_choice_weights=False, save_value_weights=False):
        self.reset()
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, self.player.num_features))
        if save_choice_weights:
            choice_weights = self.player.policy_weights.copy()
        if save_value_weights:
            value_weights = self.player.value_weights.copy()
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            if self.game_over:
                # print("Episode over")
                # print(self.ep_rewards)
                # print("len(self.ep_rewards)")
                # print(len(self.ep_rewards))
                # print("len(self.action_features)")
                # print(len(self.action_features))
                # print("len(self.actions)")
                # print(len(self.actions))
                # print("len(self.states)")
                # print(len(self.states))
                # Learn
                print("Learning now")
                self.player.learn(self.action_features, self.actions, self.states, self.ep_rewards)
                self.reset()
                print("New weights:")
                print(self.player.policy_weights)
                if save_value_weights:
                    print("New value weights:")
                    print(self.player.value_weights)
            available_actions = self.get_available_actions()
            chosen_action, action_features = self.player.choose_action(available_actions, env=self)
            self.actions.append(chosen_action)
            self.action_features.append(action_features)
            new_state, reward, game_over = self.act(chosen_action)
            self.states.append(new_state)
            self.ep_rewards.append(reward)
            self.game_over = game_over
            self.current_state = new_state
            if save_choice_weights:
                choice_weights = np.dstack((choice_weights, self.player.policy_weights.copy()))
            if save_value_weights:
                value_weights = np.dstack((value_weights, self.player.value_weights.copy()))
            if num_tests > 0 and self.cumulative_steps % test_every == 0 and self.cumulative_steps > 0:
                tested_weights[test_index] = self.player.policy_weights.copy()
                for game_ix in range(num_test_games):
                    test_results[test_index, game_ix] = test_environment.test_agent(1)
                print("Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
                test_index += 1
            self.cumulative_steps += 1
        if save_choice_weights:
            if save_value_weights:
                return test_results, choice_weights, value_weights
            else:
                return test_results, choice_weights
        else:
            return test_results, tested_weights

    def play_and_test(self, num_tests=1, num_test_games=0, test_every=0, test_environment=None,
                      save_q_trace=False, save_choice_weights=False, save_value_weights=False):
        self.reset()
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, self.player.num_features))
        if save_q_trace:
            q_trace = self.player.q_table.copy()
        if save_choice_weights:
            choice_weights = self.player.policy_weights.copy()
        if save_value_weights:
            value_weights = self.player.value_weights.copy()
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            if num_tests > 0 and self.cumulative_steps % test_every == 0:  # and self.cumulative_steps > 0
                # tested_weights[test_index] = self.player.policy_weights.copy()
                # print("tested_weights", tested_weights)
                # print("TESTING: ", test_index + 1, " out of ", num_tests, " tests.")
                for game_ix in range(num_test_games):
                    test_results[test_index, game_ix] = test_environment.test_agent(1)
                    # print("Game ", game_ix, " had ", test_results[test_index, game_ix], " cleared lines.")
                print("Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
                test_index += 1
            if self.game_over:
                self.reset()
                if self.player.name == "ActorCritic":
                    self.player.ind = 1
            available_actions = self.get_available_actions()
            chosen_action, action_features = self.player.choose_action(available_actions, env=self)
            new_state, reward, game_over = self.act(chosen_action)
            self.game_over = game_over
            if self.player.name == "Choice":
                self.player.learn(action_features, chosen_action)
            elif self.player.name == "ActorCritic":
                self.player.learn(self.current_state, reward, new_state, chosen_action, new_state_is_terminal=self.game_over,
                                  action_features=action_features)
            else:
                self.player.learn(self.current_state, reward, new_state, chosen_action, new_state_is_terminal=self.game_over)
            self.current_state = new_state
            if save_q_trace:
                q_trace = np.dstack((q_trace, self.player.q_table.copy()))
            if save_choice_weights:
                choice_weights = np.dstack((choice_weights, self.player.policy_weights.copy()))
            if save_value_weights:
                value_weights = np.dstack((value_weights, self.player.value_weights.copy()))
            self.cumulative_steps += 1
        if save_q_trace:
            return test_results, q_trace
        elif save_choice_weights:
            if save_value_weights:
                return test_results, choice_weights, value_weights
            else:
                return test_results, choice_weights
        else:
            return test_results, tested_weights

    def play(self, max_steps=np.inf, save_q_trace=False, save_choice_weights=False):
        self.reset()
        cumulative_reward = 0
        step = 0
        if save_q_trace:
            q_trace = self.player.q_table.copy()
        if save_choice_weights:
            choice_weights = self.player.policy_weights.copy()
        while not self.game_over and step < max_steps:
            available_actions = self.get_available_actions()
            chosen_action, action_features = self.player.choose_action(available_actions, env=self)
            new_state, reward, game_over = self.act(chosen_action)
            self.game_over = game_over
            if self.player.name == "Choice":
                self.player.learn(action_features, chosen_action)
            else:
                self.player.learn(self.current_state, reward, new_state, chosen_action, new_state_is_terminal=self.game_over)
            if save_q_trace:
                q_trace = np.dstack((q_trace, self.player.q_table.copy()))
            if save_choice_weights:
                choice_weights = np.dstack((choice_weights, self.player.policy_weights.copy()))
            if self.verbose:
                print("---------------------------------------------------")
                print("self.current_state", self.current_state)
                print("chosen_action", chosen_action)
                print("new_state", new_state)
                print("reward", reward)
                print("self.player.q_table", self.player.q_table)
            self.current_state = new_state
            cumulative_reward += reward
            step += 1
        if self.verbose:
            print("---------------------------------------------------")
            print("---------------------------------------------------")
            print()
            print("GAME OVER")
            print("Used steps: ", step)
            print()
            print("---------------------------------------------------")
            print("---------------------------------------------------")
        if save_q_trace:
            return cumulative_reward, q_trace
        elif save_choice_weights:
            return cumulative_reward, choice_weights
        else:
            return cumulative_reward

    def act(self, action):
        if np.random.random() < self.term_eps:
            # Go to terminal state
            new_state = ToyState(self.num_nonterminal_states, self)
            game_over = True
        else:
            game_over = False
            if action == 0:  # clockwise
                if self.current_state.state_id < self.num_nonterminal_states - 1:
                    new_state = ToyState(self.current_state.state_id + 1, self)
                else:
                    new_state = ToyState(0, self)
            elif action == 1:
                if self.current_state.state_id > 0:
                    new_state = ToyState(self.current_state.state_id - 1, self)
                else:
                    new_state = ToyState(self.num_nonterminal_states - 1, self)

            else:
                raise ValueError("Weird action choice..")
        reward = self.rewards[new_state.state_id]
        return new_state, reward, game_over

    def get_available_after_states(self):
        pass

    def get_available_actions(self):
        # 0 is
        return np.array([0, 1])


class Tetris:
    def __init__(self, num_columns, num_rows, player, verbose,
                 plot_intermediate_results=False,
                 tetromino_size=4, target_update=1, max_cleared_test_lines=np.inf):
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.tetromino_size = tetromino_size
        self.player = player
        self.verbose = verbose
        self.target_update = target_update
        self.num_features = self.player.num_features
        self.feature_type = self.player.feature_type
        self.n_fields = self.num_columns * self.num_rows
        self.game_over = False
        self.current_state = state.State(representation=np.zeros((self.num_rows + self.tetromino_size, self.num_columns), dtype=np.int_),
                                         lowest_free_rows=np.zeros(self.num_columns, dtype=np.int_), num_features=self.num_features,
                                         feature_type=self.feature_type)
        self.tetrominos = [tetromino.Straight(self.feature_type, self.num_features, self.num_columns),
                           tetromino.RCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.LCorner(self.feature_type, self.num_features, self.num_columns),
                           tetromino.Square(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeR(self.feature_type, self.num_features, self.num_columns),
                           tetromino.SnakeL(self.feature_type, self.num_features, self.num_columns),
                           tetromino.T(self.feature_type, self.num_features, self.num_columns)]
        self.tetromino_sampler = tetromino.TetrominoSamplerRandom(self.tetrominos)
        self.cleared_lines = 0
        self.state_samples = []
        self.cumulative_steps = 0
        self.max_cleared_test_lines = max_cleared_test_lines
        self.plot_intermediate_results = plot_intermediate_results

    def reset(self):
        self.game_over = False
        self.current_state = state.State(representation=np.zeros((self.num_rows + self.tetromino_size, self.num_columns), dtype=np.int_),
                                         lowest_free_rows=np.zeros(self.num_columns, dtype=np.int_), num_features=self.num_features,
                                         feature_type=self.feature_type)
        self.tetromino_sampler = tetromino.TetrominoSampler(self.tetrominos)
        self.cleared_lines = 0
        self.state_samples = []

    def test_agent(self, hard_test=False):
        self.reset()
        while not self.game_over and self.cleared_lines <= self.max_cleared_test_lines:
            current_tetromino = self.tetromino_sampler.next_tetromino()
            if hard_test:
                chosen_action, action_index = self.player.choose_action_test_hard(self.current_state, current_tetromino)
            else:
                chosen_action, action_index = self.player.choose_action_test(self.current_state, current_tetromino)
            self.cleared_lines += chosen_action.n_cleared_lines
            self.current_state = chosen_action
            self.game_over = self.current_state.terminal_state
        return self.cleared_lines

    def play_cbmpi(self, testing_time=0, num_tests=1, num_test_games=0, test_points=None, test_environment=None, hard_test=False):
        self.reset()
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, self.num_features))
        tested_value_weights = np.zeros((num_tests, self.player.num_value_features+1))
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        # while not self.game_over and test_index < num_tests:
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0

        while test_index < num_tests:
            self.player.learn()
            self.cumulative_steps += 1
            if num_tests > 0 and self.cumulative_steps in test_points:
                # print(self.cumulative_steps)
                tested_weights[test_index] = self.player.policy_weights.copy()
                tested_value_weights[test_index] = self.player.value_weights.copy()
                if self.verbose:
                    print("tested_weights", tested_weights)
                    print("tested_value_weights", tested_value_weights)
                testing_time_start = time.time()
                if self.verbose:
                    print("TESTING: ", test_index + 1, " out of ", num_tests, " tests.")
                for game_ix in range(num_test_games):
                    test_results[test_index, game_ix] = test_environment.test_agent(hard_test=hard_test)
                    if self.verbose:
                        print("Game ", game_ix, " had ", test_results[test_index, game_ix], " cleared lines.")
                print("Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
                test_index += 1
                testing_time_end = time.time()
                testing_time += testing_time_end - testing_time_start
                if self.verbose:
                    print("Testing took " + str((testing_time_end - testing_time_start) / 60) + " minutes.")
        return test_results, testing_time, tested_weights

    def learn_cbmpi(self, num_iter=1):
        self.reset()
        tested_weights = np.zeros((num_iter, self.num_features))
        tested_value_weights = np.zeros((num_iter, self.player.num_value_features + 1))
        index = 0
        while index < num_iter:
            self.player.learn()
            tested_weights[index] = self.player.policy_weights.copy()
            tested_value_weights[index] = self.player.value_weights.copy()
            print("tested_weights", tested_weights[index])
            print("tested_value_weights", tested_value_weights[index])
        return tested_weights, tested_value_weights

    def play_hierarchical_learning(self, testing_time, plots_path, plot_analysis_fc, test_every=0, num_tests=1, num_test_games=0,
                                   test_points=None, test_environment=None, episode=0, agent_ix=0, store_features=False):
        self.reset()
        # self.player.reset_agent()
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, self.num_features))
        weights_storage = np.expand_dims(self.player.policy_weights, axis=0)
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            # TEST
            if num_tests > 0 and self.cumulative_steps in test_points:  # and self.cumulative_steps > 0
            # if num_tests > 0 and self.cumulative_steps % test_every == 0:  #  and self.cumulative_steps > 0
                tested_weights[test_index] = self.player.policy_weights
                print("tested_weights", tested_weights)
                testing_time_start = time.time()
                print("TESTING: ", test_index + 1, " out of ", num_tests, " tests.")
                for game_ix in range(num_test_games):
                    test_results[test_index, game_ix] = test_environment.test_agent()
                    print("Game ", game_ix, " had ", test_results[test_index, game_ix], " cleared lines.")
                print("Mean: ", np.mean(test_results[test_index, :]), ", Median: ", np.median(test_results[test_index, :]))
                test_index += 1
                if self.plot_intermediate_results:
                    plot_analysis_fc(plots_path, tested_weights, test_results, weights_storage, agent_ix)
                testing_time_end = time.time()
                testing_time += testing_time_end - testing_time_start
            if self.game_over:
                self.reset()
            if self.verbose:
                print("Episode: ", episode, ", Step: ", self.player.step, "Cleared lines: ", self.cleared_lines)
            current_tetromino = self.tetromino_sampler.next_tetromino()
            if self.verbose:
                print(current_tetromino)
                self.current_state.print_board()
            bef = time.time()
            chosen_action, action_index, action_features = self.player.choose_action(start_state=self.current_state,
                                                                                     start_tetromino=current_tetromino)
            af = time.time()
            # if self.verbose:
            print("CURRENT STEP: " + str(self.cumulative_steps))
            print("Choosing an action took: " + str(af-bef) + " seconds.")
            self.game_over = chosen_action.terminal_state

            store_features = store_features if self.player.phase == "learn_directions" else False
            # Change state
            weights_storage = np.vstack((weights_storage, self.player.policy_weights))
            self.cleared_lines += chosen_action.n_cleared_lines
            self.current_state = chosen_action

            # LEARN
            if not self.player.ew:
                bef = time.time()
                if not self.game_over:
                    print("Started learning")
                    switched_phase = self.player.learn(action_features=action_features, action_index=action_index)
                    # if self.player.phase == "learn_directions":
                    #     pass
                    #     # self.player.push_data(action_features, action_index)
                    #     # print("Direction counts:")
                    #     # print(self.player.positive_direction_counts / self.player.meaningful_comparisons - 0.5)
                    #     # print("Learnt directions:")
                    #     # print(self.player.decided_directions)
                    # elif self.player.phase == "learn_order":
                    #     print("Current ordering")
                    #     print(np.argsort(-self.player.positive_direction_counts/self.player.meaningful_comparisons))
                    #     print("Current ratios")
                    #     print(self.player.positive_direction_counts/self.player.meaningful_comparisons)
                    # elif self.player.phase == "learn_weights":
                    #     # if self.verbose:
                    #     print("CURRENT WEIGHTS:")
                    #     print(self.player.policy_weights * self.player.feature_directors)
                    if switched_phase and self.player.phase != "optimize_weights":
                        print("RESET GAME!")
                        self.reset()
                af = time.time()
                print("Learning took: " + str(af-bef) + " seconds.")
                print("self.player.mlogit_data.choice_set_counter: " + str(self.player.mlogit_data.choice_set_counter))
                print("self.player.mlogit_data.current_number_of_choice_sets: " + str(self.player.mlogit_data.current_number_of_choice_sets))
            self.player.step += 1
            self.cumulative_steps += 1
        # if self.game_over:
        #     raise ValueError("Training game should not be over...!!")
        return self.cleared_lines, test_results, testing_time, tested_weights, weights_storage

    def store_moves(self):
        self.reset()
        while not self.game_over and self.cleared_lines <= self.max_cleared_test_lines:
            current_tetromino = self.tetromino_sampler.next_tetromino()
            if self.verbose:
                print(current_tetromino)
                self.current_state.print_board()
            # self.state_samples.append(self.current_state)
            chosen_action, action_index = self.player.choose_action_test(self.current_state, current_tetromino)
            if not chosen_action.terminal_state:
                self.state_samples.append(self.current_state)
            self.cleared_lines += chosen_action.n_cleared_lines
            self.current_state = chosen_action
            self.game_over = self.current_state.terminal_state
        return self.cleared_lines, self.state_samples

    def is_game_over(self):
        if np.any(self.current_state.representation[self.num_rows]):
            self.game_over = True


Sample = collections.namedtuple('Sample', ('state', 'tetromino'))


