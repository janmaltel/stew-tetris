from tetris import agents, game
from .utils import plot_individual_agent, plot_analysis
import numpy as np
import random
import time
import os

"""
This file 
"""

def p_loop(p, seed, plot_individual=False, return_action_count=False):
    random.seed(seed + p.seed)
    np.random.seed(seed + p.seed)
    # torch.manual_seed(seed)

    player = agents.HierarchicalLearner(start_phase_ix=p.start_phase_ix, feature_type=p.feature_type,
                                        num_columns=p.num_columns,
                                        verbose=p.verbose, verbose_stew=p.verbose_stew,
                                        learn_from_step_phases=p.learn_from_step_phases,
                                        avg_expands_per_children_phases=p.avg_expands_per_children_phases,
                                        delete_every_phases=p.delete_every_phases,
                                        lambda_min=p.lambda_min,
                                        lambda_max=p.lambda_max,
                                        num_lambdas=p.num_lambdas,
                                        gamma_phases=p.gamma_phases,
                                        rollout_length_phases=p.rollout_length_phases,
                                        rollout_action_selection_phases=p.rollout_action_selection_phases,
                                        max_length_phases=p.max_length_phases,
                                        dom_filter_phases=p.dom_filter_phases,
                                        cumu_dom_filter_phases=p.cumu_dom_filter_phases,
                                        rollout_dom_filter_phases=p.rollout_dom_filter_phases,
                                        rollout_cumu_dom_filter_phases=p.rollout_cumu_dom_filter_phases,
                                        filter_best_phases=p.filter_best_phases,
                                        ols_phases=p.ols_phases,
                                        feature_directors=p.feature_directors,
                                        standardize_features=p.standardize_features,
                                        max_batch_size=p.max_batch_size,
                                        learn_every_after=p.learn_every_after,
                                        learn_every_step_until=p.learn_every_step_until,
                                        ew=p.ew,
                                        random_init_weights=p.random_init_weights,
                                        do_sgd_update=p.do_sgd_update,
                                        ridge=p.ridge,
                                        one_se_rule=p.one_se_rule,
                                        nonnegative=p.nonnegative)

    environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=p.verbose)
    test_environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=False,
                                   max_cleared_test_lines=p.max_cleared_test_lines)

    start = time.time()
    testing_time = 0
    _, test_results_ix, testing_time, tested_weights_ix, weights_storage_ix = \
        environment.play_hierarchical_learning(plots_path=p.plots_path,
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


def cbmpi_loop(p, seed, D):
    random.seed(seed * p.seed)
    np.random.seed(seed * p.seed)
    player = agents.Cbmpi(m=p.m,
                          N=p.N,
                          M=p.M,
                          D=D,
                          B=p.B,
                          feature_type=p.feature_type,
                          num_columns=p.num_columns,
                          cmaes_var=p.cmaes_var,
                          verbose=p.verbose,
                          seed=seed,
                          id=p.seed,
                          discrete_choice=p.discrete_choice)

    environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=p.verbose)
    test_environment = game.Tetris(num_columns=p.num_columns, num_rows=p.num_rows, player=player, verbose=p.verbose,
                                   max_cleared_test_lines=p.max_cleared_test_lines)

    start = time.time()
    testing_time = 0
    test_results_ix, testing_time_ix, tested_weights_ix = environment.play_cbmpi(testing_time=testing_time,
                                                                                 num_tests=p.num_tests,
                                                                                 num_test_games=p.num_test_games,
                                                                                 test_points=p.test_points,
                                                                                 test_environment=test_environment,
                                                                                 hard_test=p.hard_test)
    end = time.time()
    total_time = end - start
    return [test_results_ix, tested_weights_ix, total_time]

