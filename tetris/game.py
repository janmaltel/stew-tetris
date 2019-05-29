import numpy as np
from tetris import state, tetromino
import collections
import time

Sample = collections.namedtuple('Sample', ('state', 'tetromino'))


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

    def play_m_learning(self, testing_time, plots_path, plot_analysis_fc, test_every=0, num_tests=1, num_test_games=0,
                        test_points=None, test_environment=None, episode=0, agent_ix=0, store_features=False):
        self.reset()
        test_results = np.zeros((num_tests, num_test_games))
        tested_weights = np.zeros((num_tests, self.num_features))
        weights_storage = np.expand_dims(self.player.policy_weights, axis=0)
        if num_tests == 0:
            test_index = -1
        else:
            test_index = 0
        while test_index < num_tests:
            # TEST
            if num_tests > 0 and self.cumulative_steps in test_points:
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
            if self.verbose:
                print("CURRENT STEP: " + str(self.cumulative_steps))
                print("Choosing an action took: " + str(af-bef) + " seconds.")
            self.game_over = chosen_action.terminal_state

            # Change state
            weights_storage = np.vstack((weights_storage, self.player.policy_weights))
            self.cleared_lines += chosen_action.n_cleared_lines
            self.current_state = chosen_action

            # LEARN
            if self.player.regularization != "ew":
                bef = time.time()
                if not self.game_over:
                    print("Started learning")
                    self.player.learn(action_features=action_features, action_index=action_index)
                af = time.time()
                print("Learning took: " + str(af-bef) + " seconds.")
                print("self.player.mlogit_data.choice_set_counter: " + str(self.player.mlogit_data.choice_set_counter))
                print("self.player.mlogit_data.current_number_of_choice_sets: " + str(self.player.mlogit_data.current_number_of_choice_sets))
            self.player.step += 1
            self.cumulative_steps += 1
        return self.cleared_lines, test_results, testing_time, tested_weights, weights_storage

    def is_game_over(self):
        if np.any(self.current_state.representation[self.num_rows]):
            self.game_over = True





