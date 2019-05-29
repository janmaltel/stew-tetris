import os
from datetime import datetime
import multiprocessing
import numpy as np
from tetris import play_loops
from tetris.utils import Bunch, plot_learning_curve


time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
name_id = "_stew"
run_id = time_id + name_id

run_id_path = os.path.join("experiments", run_id)
models_path = os.path.join(run_id_path, "models")
results_path = os.path.join(run_id_path, "results")
plots_path = os.path.join(run_id_path, "plots")
for path in [run_id_path, models_path, results_path, plots_path]:
    if not os.path.exists(path):
        os.makedirs(path)

param_dict = dict(
                  # Run specifics
                  num_agents=1,
                  test_points=(1, 3, 7),  #  20, 50, 100, 200, 300
                  num_tests=3,
                  num_test_games=10,
                  seed=251,
                  verbose=False,
                  verbose_stew=True,

                  regularization = "stew",  # can be either "stew", "ols", "ridge", or "nonnegative".
                  rollout_length=10,  # The third value is important. It's the variable m in the paper.
                  avg_expands_per_children=7,  # The third value is important. It's the variable m in the paper.
                  lambda_max=4,  # min regularization strength.
                  lambda_min=-8.0,  # max regularization strength.
                  num_lambdas=100,  # number of tested reg strengths.
                  dominance_filter=True,
                  cumu_dom_filter=True,
                  rollout_dom_filter=True,
                  rollout_cumu_dom_filter=True,
                  filter_best=False,
                  gamma=0.995,
                  delete_every=2,
                  learn_from_step=2,
                  feature_directors=None,
                  learn_periodicity=100,
                  learn_every_step_until=50,
                  max_batch_size=50,

                  # Tetris params
                  num_columns=10,
                  num_rows=10,
                  feature_type='bcts',
                  max_cleared_test_lines=200000)

param_dict["plots_path"] = plots_path
plot_individual = False
p = Bunch(param_dict)


###
###  RUN
###
ncpus = multiprocessing.cpu_count()
print("NUMBER OF CPUS: " + str(ncpus))

pool = multiprocessing.Pool(np.minimum(ncpus, p.num_agents))
results = [pool.apply_async(play_loops.m_learning_play_loop, (p, seed, plot_individual)) for seed in np.arange(p.num_agents)]

test_results = [results[ix].get()[0] for ix in np.arange(p.num_agents)]
test_results = np.stack(test_results, axis=0)
# Save test results
np.save(file=os.path.join(results_path, "test_results.npy"), arr=test_results)

###
###  PLOT LEARNING CURVE
###
plot_learning_curve(plots_path, test_results, x_axis=p.test_points)

print("Results can be found in directory: " + run_id)


