import sys
import time
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

from tetris import state, ploops
from tetris.utils import Bunch, plot_learning_curve



###
###  INIT
###

# Define parameters
param_dict = dict(
                  # Run specifics
                  num_agents=1,  # Increase this if averaging across many agents.
                  test_points=(2, 4, 10),
                  num_tests=3,
                  num_test_games=3,
                  seed=14,
                  verbose=True,

                  # CBMPI params
                  cmaes_var=1,
                  m=5,
                  B=23800,
                  N=None,
                  M=1,
                  D="rollout_sets/gabillon/record_du10++cat.txt",

                  # Tetris params
                  num_columns=10,
                  num_rows=10,
                  feature_type='bcts',
                  standardize_features=False,
                  max_cleared_test_lines=100000,

                  # Additional params (Unimportant here / from previous versions)
                  # IGNORE
                  num_episodes=1,
                  verbose_stew=False,
                  hard_test=False,
                  test_every=60,
                  discrete_choice=False,
                  )

# Define save paths and create directories
time_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
run_id = time_id + "_cbmpi_23800"

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

param_dict["plots_path"] = plots_path

p = Bunch(param_dict)

# Load rollout data set from Scherrer et al. (2015)
sample_list_save_name = p.D
with open(sample_list_save_name, "r") as ins:
    D = [state.State(representation=np.vstack((np.array([np.array([int(z) for z in bin(int(y))[3:13]]) for y in x.split()]),
                                               np.zeros((4, p.num_columns)))))
         for x in ins]


###
###  RUN
###
start_time = time.time()
pool = multiprocessing.Pool(np.minimum(multiprocessing.cpu_count() - 1, p.num_agents))
results = [pool.apply_async(ploops.cbmpi_loop, (p, seed, D)) for seed in np.arange(p.num_agents)]
end_time = time.time()

test_results = [results[ix].get()[0] for ix in np.arange(p.num_agents)]
test_results = np.stack(test_results, axis=0)


###
###  PLOT LEARNING CURVE
###

plot_learning_curve(plots_path, test_results, x_axis=np.arange(p.num_tests))

# plot_analysis(plots_path, tested_weights, test_results, weights_storage, agent_ix="MEAN")

print("Results can be found in directory: " + run_id)

with open(os.path.join(run_id_path, "Info.txt"), "w") as text_file:
    print("Started at: " + time_id, file=text_file)  # + " from file " + str(__file__)
    print("Time spent: " + str((end_time-start_time)/3600) + "hours.")

