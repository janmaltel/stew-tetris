import sys
import os
from datetime import datetime
import multiprocessing

try:
    import numpy as np
except ImportError:
    sys.exit("""You need numpy!
                install it using
                pip install numpy
                in the command line""")
try:
    import matplotlib
except ImportError:
    sys.exit("""You need matplotlib!
                install it using
                pip install matplotlib
                in the command line""")

try:
    import numba
except ImportError:
    sys.exit("""You need numba!
                install it using
                pip install numba
                in the command line""")

try:
    import scipy
except ImportError:
    sys.exit("""You need scipy!
                install it using
                pip install scipy
                in the command line""")

try:
    import sklearn
except ImportError:
    sys.exit("""You need sklearn!
                install it using
                pip install sklearn
                in the command line""")

try:
    import cma
except ImportError:
    sys.exit("""You need cma!
                install it using
                pip install cma
                in the command line""")

from tetris import ploops
from tetris.utils import Bunch, plot_learning_curve


###
###  INIT
###
time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
name_id = "_stew"
run_id = time_id + name_id

run_id_path = os.path.join("experiments", run_id)
if not os.path.exists(run_id_path):
    os.makedirs(run_id_path)
# model_save_name = os.path.join(dir_path, "model.pt")

models_path = os.path.join(run_id_path, "models")
if not os.path.exists(models_path):
    os.makedirs(models_path)

results_path = os.path.join(run_id_path, "results")
if not os.path.exists(results_path):
    os.makedirs(results_path)

plots_path = os.path.join(run_id_path, "plots")
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

param_dict = dict(
                  # Run specifics
                  num_agents=1,
                  test_points=(1, 3, 7),  #  20, 50, 100, 200, 300
                  num_tests=3,
                  num_test_games=10,
                  seed=251,
                  verbose=False,
                  verbose_stew=True,

                  # STEW algorithm params (If the following two params are all False,
                  # algo will use STEW, otherwise turn either ols or ridge or ew (=equal weights)
                  # to True, not 2 at the same time!
                  ols_phases=(False, False, False, False),
                  ridge=False,
                  ew=False,
                  rollout_length_phases=(20, 50, 10, 100),  # The third value is important. It's the variable m in the paper.
                  avg_expands_per_children_phases=(20, 15, 7, 10),  # The third value is important. It's the variable m in the paper.
                  lambda_max=4,  # min regularization strength.
                  lambda_min=-8.0,  # max regularization strength.
                  num_lambdas=100,  # number of tested reg strengths.

                  # Tetris params
                  num_columns=10,
                  num_rows=10,
                  feature_type='bcts',
                  standardize_features=False,
                  max_cleared_test_lines=200000,

                  test_every=20,
                  do_sgd_update=False,
                  hard_test=False,

                  # Additional params. Not important for here.
                  # IGNORE
                  start_phase_ix=2,
                  max_length_phases=(50, 50, np.inf, np.inf),
                  dom_filter_phases=(False, True, True, True),
                  cumu_dom_filter_phases=(False, False, True, True),
                  rollout_dom_filter_phases=(False, True, True, False),
                  rollout_cumu_dom_filter_phases=(False, False, True, False),
                  filter_best_phases=(False, False, False, False),
                  gamma_phases=(0.8, 0.9, 0.995, 0.995),
                  rollout_action_selection_phases=("random", "random", "max_util", "max_util"),
                  delete_every_phases=(0, 0, 2, 0),
                  learn_from_step_phases=(0, 0, 2, 0),
                  feature_directors=None,
                  learn_every_after=100,
                  learn_every_step_until=50,
                  num_episodes=1,
                  random_init_weights=True,
                  max_batch_size=50,
                  one_se_rule=False,
                  nonnegative=False,
                  notify=False)

param_dict["plots_path"] = plots_path
plot_individual = False
p = Bunch(param_dict)


###
###  RUN
###
ncpus = multiprocessing.cpu_count()
print("NUMBER OF CPUS: " + str(ncpus))

pool = multiprocessing.Pool(np.minimum(ncpus, p.num_agents))
results = [pool.apply_async(ploops.p_loop, (p, seed, plot_individual)) for seed in np.arange(p.num_agents)]

test_results = [results[ix].get()[0] for ix in np.arange(p.num_agents)]
test_results = np.stack(test_results, axis=0)
# Save test results
np.save(file=os.path.join(results_path, "test_results.npy"), arr=test_results)

###
###  PLOT LEARNING CURVE
###
plot_learning_curve(plots_path, test_results, x_axis=p.test_points)

print("Results can be found in directory: " + run_id)


