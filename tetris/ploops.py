from tetris import game
import mLearning
from .utils import plot_individual_agent, plot_analysis
import numpy as np
import random
import time
import os

"""
This file 
"""

def m_learning_play_loop(p, seed, plot_individual=False, return_action_count=False):
    random.seed(seed + p.seed)
    np.random.seed(seed + p.seed)
    player = mLearning.MLearning(feature_type=p.feature_type,
                                 num_columns=p.num_columns,
                                 verbose=p.verbose, verbose_stew=p.verbose_stew,
                                 avg_expands_per_children=p.avg_expands_per_children,
                                 lambda_min=p.lambda_min,
                                 lambda_max=p.lambda_max,
                                 num_lambdas=p.num_lambdas,
                                 gamma=p.gamma,
                                 rollout_length=p.rollout_length,
                                 dominance_filter=p.dominance_filter,
                                 cumu_dom_filter=p.cumu_dom_filter,
                                 rollout_dom_filter=p.rollout_dom_filter,
                                 rollout_cumu_dom_filter=p.rollout_cumu_dom_filter,
                                 regularization=p.regularization,
                                 feature_directors=p.feature_directors,
                                 max_batch_size=p.max_batch_size,
                                 learn_periodicity=p.learn_periodicity,
                                 learn_every_step_until=p.learn_every_step_until
                                 )

    environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=p.verbose)
    test_environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=False,
                                   max_cleared_test_lines=p.max_cleared_test_lines)

    start = time.time()
    testing_time = 0
    _, test_results_ix, testing_time, tested_weights_ix, weights_storage_ix = \
        environment.play_m_learning(plots_path=p.plots_path,
                                    plot_analysis_fc=plot_analysis,
                                    test_every=p.test_every,
                                    num_tests=p.num_tests,
                                    num_test_games=p.num_test_games,
                                    test_points=p.test_points,
                                    test_environment=test_environment,
                                    testing_time=testing_time,
                                    agent_ix=seed,
                                    store_features=True)
    end = time.time()
    total_time = end - start
    print("TESTING TIME " + str(testing_time / 3600) + " hours.")
    print("TOTAL TIME " + str(total_time / 3600) + " hours.")

    if plot_individual:
        plots_path_ind = os.path.join(p.plots_path, "agent" + str(seed))
        if not os.path.exists(plots_path_ind):
            os.makedirs(plots_path_ind, exist_ok=True)
        plot_individual_agent(plots_path=plots_path_ind, tested_weights=tested_weights_ix, test_results=test_results_ix,
                              weights_storage=weights_storage_ix, agent_ix=seed, x_axis=p.test_points)

    if return_action_count:
        return [test_results_ix, tested_weights_ix, weights_storage_ix, testing_time / 3600, player.gen_model_count]
    else:
        return [test_results_ix, tested_weights_ix, weights_storage_ix, testing_time/3600]

