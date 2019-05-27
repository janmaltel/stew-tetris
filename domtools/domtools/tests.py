import domtools.domfilter as df
import numpy as np

test_features = np.array([[2, 2, 2, 2, 2],  # 0 dom
                          [0, 1, 2, 3, 4],  # 1 cum dom
                          [4, 3, 2, 1, 0],  # 2 dom
                          [3, 2, 1, 0, -1], # 3 dom
                          [5, 4, 3, 2, 1],  # 4
                          [3, 3, 3, 3, 3],  # 6 cum dom
                          [3, 3, 3, 3, 0],  # 7 dom
                          [3, 2, 4, 3, 3],  # 8 cum dom
                          [0, 4, 3, 3, 3]]) # 9 cum dom
len_after_states = len(test_features)

# test_features = np.array([
#                           # [2, 2, 2, 2, 2],  # 0 dom
#                           # [0, 1, 2, 3, 4],  # 1 cum dom
#                           [4, 3, 2, 1, 0],  # 2 dom
#                           [3, 2, 1, 0, -1] # 3 dom
#                           # [5, 4, 3, 2, 1],  # 4
#                           # [3, 3, 3, 3, 3],  # 6 cum dom
#                           # [3, 3, 3, 3, 0],  # 7 dom
#                           # [3, 2, 4, 3, 3],  # 8 cum dom
#                           # [0, 4, 3, 3, 3]
# ]) # 9 cum dom
# len_after_states = len(test_features)

# test_features = np.array([[3, 3, 3, 3, 3],  #
#                           [3, 2, 4, 3, 3]])  # 9 cum dom
# len_after_states = len(test_features)

sim_1, cumu_1 = df.inner_loop(test_features, len_after_states)
ap_sim_1, ap_cumu_1, ap_sim_2, ap_cumu_2 = df.approx_inner_loop(test_features, len_after_states, threshold=0)

print(np.invert(sim_1))
print(np.invert(ap_sim_1))
print(np.invert(ap_sim_2))
print(np.invert(cumu_1))
print(np.invert(ap_cumu_1))
print(np.invert(ap_cumu_2))

np.testing.assert_array_equal(sim_1, ap_sim_1)
np.testing.assert_array_equal(ap_sim_1, ap_sim_2)

np.testing.assert_array_equal(cumu_1, ap_cumu_1)
np.testing.assert_array_equal(ap_cumu_1, ap_cumu_2)
