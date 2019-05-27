from numba import njit
import numpy as np


@njit
def dom_rel(f_i, f_j):
    """
    Calculate dominance relationship (simple and cumulative) between two feature arrays.

    :param f_i: array like, feature array of alternative i
    :param f_j: array like, feature array of alternative j
    :return (out_simple, out_cumu): where
        out_simple: integer, 1 if f_i simply dominates f_j, -1 if f_j simply dominates f_i, and 0 otherwise
        out_cumu: integer, 1 if f_i cumulatively dominates f_j, -1 if f_j cumulatively dominates f_i, and 0 otherwise
    """
    i_dominates_j = True
    j_dominates_i = True
    strictly_larger = False
    strictly_smaller = False
    i_cumu_dominates_j = True
    j_cumu_dominates_i = True
    cumu_strictly_larger = False
    cumu_strictly_smaller = False

    diff_cumu = 0
    for ix in range(len(f_i)):
        diff = f_i[ix] - f_j[ix]
        diff_cumu += diff
        if diff > 0:
            j_dominates_i = False
            strictly_larger = True
        elif diff < 0:
            i_dominates_j = False
            strictly_smaller = True
        if diff_cumu > 0:
            j_cumu_dominates_i = False
            cumu_strictly_larger = True
        elif diff_cumu < 0:
            i_cumu_dominates_j = False
            cumu_strictly_smaller = True
        if not j_cumu_dominates_i and not i_cumu_dominates_j:
            return 0, 0
    if i_cumu_dominates_j and cumu_strictly_larger:
        out_cumu = 1
    elif j_cumu_dominates_i and cumu_strictly_smaller:
        out_cumu = -1
    else:
        out_cumu = 0
    if i_dominates_j and strictly_larger:
        out_simple = 1
    elif j_dominates_i and strictly_smaller:
        out_simple = -1
    else:
        out_simple = 0
    return out_simple, out_cumu


@njit
def approx_dom_rel(f_i, f_j, threshold):
    """
    Calculate approximate dominance relationship (simple and cumulative) between two feature arrays.

    :param f_i: array like, feature array of alternative i
    :param f_j: array like, feature array of alternative j
    :param threshold: integer, number of allowed counter-examples
    :return (out_simple, out_cumu): where
        out_simple: integer, 1 if f_i simply dominates f_j, -1 if f_j simply dominates f_i, and 0 otherwise
        out_cumu: integer, 1 if f_i cumulatively dominates f_j, -1 if f_j cumulatively dominates f_i, and 0 otherwise
    """
    i_dominates_j = True
    j_dominates_i = True
    i_dominates_j_count = 0
    j_dominates_i_count = 0
    strictly_larger = False
    strictly_smaller = False
    i_cumu_dominates_j = True
    j_cumu_dominates_i = True
    i_cumu_dominates_j_count = 0
    j_cumu_dominates_i_count = 0
    cumu_strictly_larger = False
    cumu_strictly_smaller = False

    diff_cumu = 0
    for ix in range(len(f_i)):
        diff = f_i[ix] - f_j[ix]
        diff_cumu += diff
        if diff > 0:
            j_dominates_i = False
            strictly_larger = True
            i_dominates_j_count += 1
        elif diff < 0:
            i_dominates_j = False
            strictly_smaller = True
            j_dominates_i_count += 1
        if diff_cumu > 0:
            j_cumu_dominates_i = False
            cumu_strictly_larger = True
            i_cumu_dominates_j_count += 1
        elif diff_cumu < 0:
            i_cumu_dominates_j = False
            cumu_strictly_smaller = True
            j_cumu_dominates_i_count += 1
        if j_cumu_dominates_i_count > threshold and i_cumu_dominates_j_count > threshold:
            return 0, 0, 0, 0
        # if not j_dominates_i and not i_dominates_j:
        #     return 0
    if i_cumu_dominates_j and cumu_strictly_larger:
        out_cumu = 1
    elif j_cumu_dominates_i and cumu_strictly_smaller:
        out_cumu = -1
    else:
        out_cumu = 0

    if i_dominates_j and strictly_larger:
        out_simple = 1
    elif j_dominates_i and strictly_smaller:
        out_simple = -1
    else:
        out_simple = 0

    if j_dominates_i_count <= threshold and i_dominates_j_count <= threshold:
        out_simple_approx = 0
    elif j_dominates_i_count <= threshold:
        out_simple_approx = 1
    elif i_dominates_j_count <= threshold:
        out_simple_approx = -1
    else:
        out_simple_approx = 0

    if j_cumu_dominates_i_count <= threshold and i_cumu_dominates_j_count <= threshold:
        out_cumu_approx = 0
    elif j_cumu_dominates_i_count <= threshold:
        out_cumu_approx = 1
    elif i_cumu_dominates_j_count <= threshold:
        out_cumu_approx = -1
    else:
        out_cumu_approx = 0

    return out_simple, out_cumu, out_simple_approx, out_cumu_approx


@njit
def inner_loop(features, len_features):
    """
    Takes a set of alternatives (described by features) and calculates, which
    alternatives (rows) are NOT simply or cumulatively dominated.

    :param features: two dimensional numpy array, rows are alternatives, columns are features.
    :return (not_simply_dominated, not_cumu_dominated): where
        not_simply_dominated boolean mask numpy array, alternatives that are not simply dominated
        not_cumu_dominated boolean mask numpy array, alternatives that are not cumulatively dominated
    """
    len_features = len(features)
    not_simply_dominated = np.ones(len_features, dtype=np.bool_)
    not_cumu_dominated = np.ones(len_features, dtype=np.bool_)
    for after_state_ix in range(len_features):
        if not_simply_dominated[after_state_ix]:
            range_of_j = np.arange(after_state_ix + 1, len_features)[not_simply_dominated[after_state_ix + 1:]]
            for after_state_jx in range_of_j:
                simply, cumu = dom_rel(features[after_state_ix], features[after_state_jx])
                if cumu == 1:
                    # print("Line ", after_state_ix, " cumu dominates line ", after_state_jx)
                    not_cumu_dominated[after_state_jx] = False
                elif cumu == -1:
                    # print("Line ", after_state_jx, " cumu dominates line ", after_state_ix)
                    not_cumu_dominated[after_state_ix] = False
                if simply == 1:
                    # print("Line ", after_state_ix, " simply dominates line ", after_state_jx)
                    not_simply_dominated[after_state_jx] = False
                elif simply == -1:
                    # print("Line ", after_state_jx, " simply dominates line ", after_state_ix)
                    not_simply_dominated[after_state_ix] = False
                    break
    return not_simply_dominated, not_cumu_dominated


@njit
def dom_filter(features, len_after_states):
    not_simply_dominated = np.ones(len_after_states, dtype=np.bool_)
    not_cumu_dominated = np.ones(len_after_states, dtype=np.bool_)
    for after_state_ix in range(len_after_states):
        if not_simply_dominated[after_state_ix]:
            range_of_j = np.arange(after_state_ix + 1, len_after_states)[not_simply_dominated[after_state_ix + 1:]]
            for after_state_jx in range_of_j:
                simply, cumu = dom_rel(features[after_state_ix], features[after_state_jx])
                if cumu == 1:
                    # print("Line ", after_state_ix, " cumu dominates line ", after_state_jx)
                    not_cumu_dominated[after_state_jx] = False
                elif cumu == -1:
                    # print("Line ", after_state_jx, " cumu dominates line ", after_state_ix)
                    not_cumu_dominated[after_state_ix] = False
                if simply == 1:
                    # print("Line ", after_state_ix, " simply dominates line ", after_state_jx)
                    not_simply_dominated[after_state_jx] = False
                elif simply == -1:
                    # print("Line ", after_state_jx, " simply dominates line ", after_state_ix)
                    not_simply_dominated[after_state_ix] = False
                    break
    return not_simply_dominated, not_cumu_dominated


@njit
def dom_and_terminal_filter(features, len_after_states):
    not_simply_dominated = np.ones(len_after_states, dtype=np.bool_)
    not_cumu_dominated = np.ones(len_after_states, dtype=np.bool_)
    for after_state_ix in range(len_after_states):
        if not_simply_dominated[after_state_ix]:
            range_of_j = np.arange(after_state_ix + 1, len_after_states)[not_simply_dominated[after_state_ix + 1:]]
            for after_state_jx in range_of_j:
                simply, cumu = dom_rel(features[after_state_ix], features[after_state_jx])
                if cumu == 1:
                    # print("Line ", after_state_ix, " cumu dominates line ", after_state_jx)
                    not_cumu_dominated[after_state_jx] = False
                elif cumu == -1:
                    # print("Line ", after_state_jx, " cumu dominates line ", after_state_ix)
                    not_cumu_dominated[after_state_ix] = False
                if simply == 1:
                    # print("Line ", after_state_ix, " simply dominates line ", after_state_jx)
                    not_simply_dominated[after_state_jx] = False
                elif simply == -1:
                    # print("Line ", after_state_jx, " simply dominates line ", after_state_ix)
                    not_simply_dominated[after_state_ix] = False
                    break
    return not_simply_dominated, not_cumu_dominated


@njit
def approx_inner_loop(features, len_after_states, threshold):
    not_simply_dominated = np.ones(len_after_states, dtype=np.bool_)
    not_cumu_dominated = np.ones(len_after_states, dtype=np.bool_)
    not_approx_simply_dominated = np.ones(len_after_states, dtype=np.bool_)
    not_approx_cumu_dominated = np.ones(len_after_states, dtype=np.bool_)
    for after_state_ix in range(len_after_states):
        if not_simply_dominated[after_state_ix]:
            range_of_j = np.arange(after_state_ix + 1, len_after_states)[not_simply_dominated[after_state_ix + 1:]]
            for after_state_jx in range_of_j:
                simply, cumu, simply_approx, cumu_approx = approx_dom_rel(features[after_state_ix], features[after_state_jx],
                                                                          threshold)

                if simply_approx == 1:
                    # print("Line ", after_state_ix, " cumu dominates line ", after_state_jx)
                    not_approx_simply_dominated[after_state_jx] = False
                elif simply_approx == -1:
                    # print("Line ", after_state_jx, " cumu dominates line ", after_state_ix)
                    not_approx_simply_dominated[after_state_ix] = False

                if cumu_approx == 1:
                    # print("Line ", after_state_ix, " cumu dominates line ", after_state_jx)
                    not_approx_cumu_dominated[after_state_jx] = False
                elif cumu_approx == -1:
                    # print("Line ", after_state_jx, " cumu dominates line ", after_state_ix)
                    not_approx_cumu_dominated[after_state_ix] = False

                if cumu == 1:
                    # print("Line ", after_state_ix, " cumu dominates line ", after_state_jx)
                    not_cumu_dominated[after_state_jx] = False
                elif cumu == -1:
                    # print("Line ", after_state_jx, " cumu dominates line ", after_state_ix)
                    not_cumu_dominated[after_state_ix] = False

                if simply == 1:
                    # print("Line ", after_state_ix, " simply dominates line ", after_state_jx)
                    not_simply_dominated[after_state_jx] = False
                elif simply == -1:
                    # print("Line ", after_state_jx, " simply dominates line ", after_state_ix)
                    not_simply_dominated[after_state_ix] = False
                    break
    return not_simply_dominated, not_cumu_dominated, not_approx_simply_dominated, not_approx_cumu_dominated
