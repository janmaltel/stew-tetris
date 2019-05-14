import numpy as np
from numba import njit
# import torch


class State:
    def __init__(self, representation, lowest_free_rows=None,
                 anchor_col=0,
                 changed_lines=np.arange(1),
                 pieces_per_changed_row=np.array([0]),
                 landing_height_bonus=0.0,
                 num_features=8,
                 feature_type='bcts'):
        self.representation = representation
        if lowest_free_rows is None:
            # raise ValueError("Should not calc_lowest_free_rows.")
            self.lowest_free_rows = calc_lowest_free_rows(self.representation)
        else:
            self.lowest_free_rows = lowest_free_rows
        self.anchor_col = anchor_col
        self.num_rows = self.representation.shape[0]
        self.num_columns = self.representation.shape[1]
        self.pieces_per_changed_row = pieces_per_changed_row
        self.landing_height_bonus = landing_height_bonus
        self.num_features = num_features
        self.feature_type = feature_type

        self.n_legal_rows = self.num_rows - 4
        self.n_cleared_lines = 0
        self.anchor_row = changed_lines[0]
        self.cleared_rows_relative_to_anchor = self.clear_lines(changed_lines=changed_lines)

        self.features = None
        # self.calc_feature_values()
        self.terminal_state = check_terminal(self.representation, self.n_legal_rows)  # self.is_terminal()
        # TODO: change reward back to -10??
        self.reward = 0 if self.terminal_state else self.n_cleared_lines
        # self.reward = 0 if self.terminal_state else self.n_cleared_lines
        self.value_estimate = 0.0

    def __repr__(self):
        return self.print_board_to_string()

    def get_features(self, direct_by=None, order_by=None, standardize_by=None, addRBF=False):
        if self.features is None:
            self.calc_feature_values()
        features = self.features
        if order_by is not None:
            features = features[order_by]
        if direct_by is not None:
            features = features * direct_by
        if standardize_by is not None:
            features = features / standardize_by
        if addRBF:
            features = np.append(features, np.exp(-(np.mean(self.lowest_free_rows) - np.arange(5) * self.n_legal_rows / 4)**2 / (2*(self.n_legal_rows / 5)**2)))
        return features

    # TODO: Implement order / directions...
    # def get_features_with_intercept(self):
    #     if self.features is None:
    #         self.calc_feature_values()
    #     return np.insert(self.features, obj=0, values=1.)

    def print_board(self):
        for row_ix in range(self.n_legal_rows):
            # Start from top
            row_ix = self.n_legal_rows - row_ix - 1
            print("|", end=' ')
            for col_ix in range(self.num_columns):
                if self.representation[row_ix, col_ix]:
                    print("██", end=' ')
                else:
                    print("  ", end=' ')
            print("|")

    def print_board_to_string(self):
        string = "\n"
        for row_ix in range(self.n_legal_rows):
            # Start from top
            row_ix = self.n_legal_rows - row_ix - 1
            string += "|"
            for col_ix in range(self.num_columns):
                if self.representation[row_ix, col_ix]:
                    string += "██"
                else:
                    string += "  "
            string += "|\n"
        return string

    def clear_lines(self, changed_lines):
        is_full, self.n_cleared_lines, self.representation, self.lowest_free_rows = clear_lines_jitted(changed_lines,
                                                                                                       self.representation,
                                                                                                       self.lowest_free_rows,
                                                                                                       self.num_columns)
        return is_full

    def calc_feature_values(self):
        if self.feature_type == 'super_simple':
            self.calc_super_simple_features()
        elif self.feature_type == 'bcts':
            self.calc_bcts_features()
        # elif self.feature_type == "adjusted_bcts":
        #     self.calc_bcts_features(standardize_by=self.feature_stds)
        elif self.feature_type == 'simple':
            self.calc_simple_features()
        elif self.feature_type == "standardized_bcts":
            self.calc_standardized_bcts_features()

    def calc_bcts_features(self):
        features = np.zeros(self.num_features, dtype=np.float32)
        eroded_pieces = np.sum(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
        n_cleared_lines = np.sum(self.cleared_rows_relative_to_anchor)
        features[6] = eroded_pieces * n_cleared_lines
        features[3] = self.anchor_row + self.landing_height_bonus
        features[[0, 1, 2, 4, 5, 7]] = get_feature_values_jitted(lowest_free_rows=self.lowest_free_rows,
                                                                 representation=self.representation,
                                                                 num_rows=self.n_legal_rows,
                                                                 num_columns=self.num_columns)
        # ['rows_with_holes', 'column_transitions', 'holes', 'landing_height', 'cumulative_wells',
        #                  'row_transitions', 'eroded', 'hole_depth']
        # self.features = features / np.array([2.18246089, 4.42735771, 3.0698914, 2.31688581, 3.1093846, 4.0334024, 0.46720078, 8.35394364])
        self.features = features

    def calc_standardized_bcts_features(self, convert_to_numpy=True, standardize=True):
        features = np.zeros(self.num_features, dtype=np.float32)
        eroded_pieces = np.sum(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
        n_cleared_lines = np.sum(self.cleared_rows_relative_to_anchor)
        features[0] = eroded_pieces * n_cleared_lines
        features[2] = self.anchor_row + self.landing_height_bonus
        features[[1, 3, 4, 5, 6, 7]] = get_feature_values_jitted(lowest_free_rows=self.lowest_free_rows,
                                                                 representation=self.representation,
                                                                 num_rows=self.n_legal_rows,
                                                                 num_columns=self.num_columns)
        # ['eroded', 'rows_with_holes', 'landing_height', 'column_transitions', 'holes', 'cumulative_wells',
        #  'row_transitions', 'hole_depth']
        self.features = features / np.array([4.0334024, 2.18246089, 2.31688581, 4.42735771, 3.0698914, 3.1093846, 0.46720078, 8.35394364])

    # def calc_simple_features(self, convert_to_numpy=False):
    #     features = np.zeros(self.num_features, dtype=np.float32)
    #     features[:] = get_relevant_holes_jitted4(lowest_free_rows=self.lowest_free_rows,
    #                                              representation=self.representation,
    #                                              anchor_row=self.anchor_row,
    #                                              landing_height_bonus=self.landing_height_bonus,
    #                                              num_rows=self.n_legal_rows,
    #                                              num_columns=self.num_columns)
    #     # holes, cumulative_wells, cumulative_wells_squared, landing_height, avg_free_row, avg_free_row_squared
    #     # features[1] = self.anchor_row - np.min(self.lowest_free_rows) #  self.landing_height_bonus
    #     # features[2] = np.mean(self.lowest_free_rows) ** 2 #  self.landing_height_bonus
    #     self.features = torch.from_numpy(features).unsqueeze(0)
    #     if convert_to_numpy:
    #         self.features = self.features.numpy().flatten()

    def calc_super_simple_features(self, convert_to_numpy=False):
        features = np.zeros(self.num_features, dtype=np.float32)
        features[[0, 2, 3]] = get_super_simple_jitted(lowest_free_rows=self.lowest_free_rows,
                                              representation=self.representation,
                                              num_columns=self.num_columns)
        eroded_pieces = np.sum(self.cleared_rows_relative_to_anchor * self.pieces_per_changed_row)
        features[1] = eroded_pieces
        self.features = features
        # self.features = torch.from_numpy(features).unsqueeze(0)
        # if convert_to_numpy:
        #     self.features = self.features.numpy().flatten()


@njit
def check_terminal(representation, n_legal_rows):
    is_terminal = False
    for ix in representation[n_legal_rows]:
        if ix:
            is_terminal = True
            break
    return is_terminal


@njit
def clear_lines_jitted(changed_lines, representation, lowest_free_rows, num_columns):
    row_sums = np.sum(representation[changed_lines, :], axis=1)
    is_full = row_sums == num_columns
    full_lines = np.where(is_full)[0]
    n_cleared_lines = len(full_lines)
    if n_cleared_lines > 0:
        lines_to_clear = changed_lines[full_lines]
        mask_keep = np.ones(len(representation), dtype=np.bool_)
        mask_keep[lines_to_clear] = False
        representation = np.vstack((representation[mask_keep],
                                    np.zeros((n_cleared_lines, num_columns), dtype=np.int_)))
        for col_ix in range(num_columns):  # col_ix = 0
            old_lowest_free_row = lowest_free_rows[col_ix]
            if old_lowest_free_row > lines_to_clear[-1] + 1:
                lowest_free_rows[col_ix] -= n_cleared_lines
            else:
                lowest = 0
                for row_ix in range(old_lowest_free_row - n_cleared_lines - 1, -1, -1):
                    if representation[row_ix, col_ix] == 1:
                        lowest = row_ix + 1
                        break
                lowest_free_rows[col_ix] = lowest
    return is_full, n_cleared_lines, representation, lowest_free_rows


@njit
def minmaxavg_jitted(x):
    maximum = x[0]
    minimum = x[0]
    summed = 0
    for i in x[1:]:
        summed += i
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    summed = summed / len(x)
    return minimum, maximum, summed


@njit
def calc_lowest_free_rows(rep):
    num_rows, n_cols = rep.shape
    lowest_free_rows = np.zeros(n_cols, dtype=np.int_)
    for col_ix in range(n_cols):
        lowest = 0
        for row_ix in range(num_rows - 1, -1, -1):
            if rep[row_ix, col_ix] == 1:
                lowest = row_ix + 1
                break
        lowest_free_rows[col_ix] = lowest
    return lowest_free_rows


@njit
def get_relevant_holes_jitted4(lowest_free_rows, representation, anchor_row, landing_height_bonus, num_rows, num_columns):
    holes = 0.0
    cumulative_wells = 0.0
    min_lowest_free_row, max_lowest_free_row, avg_free_row = minmaxavg_jitted(lowest_free_rows)
    landing_height = anchor_row + landing_height_bonus - min_lowest_free_row
    # n_landing_positions = lowest_free_rows[1:] - lowest_free_rows[:-1]
    diffs = lowest_free_rows[1:] - lowest_free_rows[:-1]
    n_landing_positions = len(set(diffs[(-2 < diffs) & (diffs < 2)]))
    # min_lowest_free_row = np.maximum(np.min(lowest_free_rows)-3, 0)
    # max_lowest_free_row = np.max(lowest_free_rows)
    for col_ix, lowest_free_row in enumerate(lowest_free_rows):
        # col = representation[min_lowest_free_row:lowest_free_row, col_ix]

        col = representation[:lowest_free_row, col_ix]
        for row_ix, cell in enumerate(col):
            if cell == 0:
                # Holes
                holes += 1 * (0.7 + row_ix / 6)

        if col_ix == 0:
            max_well_possibility = lowest_free_rows[col_ix + 1]
        elif col_ix == num_columns - 1:
            max_well_possibility = lowest_free_rows[col_ix - 1]
        else:
            lowest_free_row_left = lowest_free_rows[col_ix - 1]
            lowest_free_row_right = lowest_free_rows[col_ix + 1]
            max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)
        local_well_streak = 0
        if max_well_possibility > lowest_free_row:
            for row_ix in range(lowest_free_row, max_well_possibility):
                if col_ix == 0:
                    cell_left = 1
                    cell_right = representation[row_ix, col_ix + 1]
                elif col_ix == num_columns - 1:
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = 1
                else:
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = representation[row_ix, col_ix + 1]

                if cell_left and cell_right:
                    local_well_streak += 1
                    cumulative_wells += local_well_streak
                    # cumulative_wells += 1
                # else:
                #     local_well_streak = 0
    cumulative_wells = 1 + cumulative_wells / 10
    avg_free_row = 1 + avg_free_row / 10
    cumulative_wells_squared = cumulative_wells ** 2
    avg_free_row_squared = avg_free_row ** 2
    features = [holes, cumulative_wells, cumulative_wells_squared,
                landing_height, avg_free_row, avg_free_row_squared, n_landing_positions]
    return features


@njit
def get_super_simple_jitted(lowest_free_rows, representation, num_columns):  # anchor_row, landing_height_bonus, num_rows,
    holes = 0.0
    cumulative_wells = 0.0
    # min_lowest_free_row, max_lowest_free_row, avg_free_row = minmaxavg_jitted(lowest_free_rows)
    # landing_height = anchor_row + landing_height_bonus - min_lowest_free_row
    diffs = lowest_free_rows[1:] - lowest_free_rows[:-1]
    n_landing_positions = len(set(diffs[(-2 < diffs) & (diffs < 2)]))
    for col_ix, lowest_free_row in enumerate(lowest_free_rows):
        col = representation[:lowest_free_row, col_ix]
        local_well_streak = 0
        for row_ix, cell in enumerate(col):
            if cell == 0:
                # Holes
                holes += 1 * (0.8 + row_ix / 8)
                # holes += 1

                # Count capped wells as well!
                if col_ix == 0:
                    cell_left = 1
                    cell_right = representation[row_ix, col_ix + 1]
                elif col_ix == num_columns - 1:
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = 1
                else:
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = representation[row_ix, col_ix + 1]

                if cell_left and cell_right:
                    local_well_streak += 1
                    cumulative_wells += local_well_streak
                else:
                    local_well_streak = 0

        if col_ix == 0:
            max_well_possibility = lowest_free_rows[col_ix + 1]
        elif col_ix == num_columns - 1:
            max_well_possibility = lowest_free_rows[col_ix - 1]
        else:
            lowest_free_row_left = lowest_free_rows[col_ix - 1]
            lowest_free_row_right = lowest_free_rows[col_ix + 1]
            max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)
        local_well_streak = 0
        if max_well_possibility > lowest_free_row:
            for row_ix in range(lowest_free_row, max_well_possibility):
                if col_ix == 0:
                    cell_left = 1
                    cell_right = representation[row_ix, col_ix + 1]
                elif col_ix == num_columns - 1:
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = 1
                else:
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = representation[row_ix, col_ix + 1]

                if cell_left and cell_right:
                    local_well_streak += 1
                    cumulative_wells += local_well_streak
                else:
                    local_well_streak = 0
    features = [holes, cumulative_wells/5, n_landing_positions]
    return features


@njit
def get_feature_values_jitted(lowest_free_rows, representation, num_rows, num_columns):
    rows_with_holes_set = {100}
    column_transitions = 0
    holes = 0
    # landing_height
    cumulative_wells = 0
    row_transitions = 0
    # eroded_piece_cells
    hole_depth = 0
    for col_ix, lowest_free_row in enumerate(lowest_free_rows):
        if col_ix == 0:
            local_well_streak = 0
            if lowest_free_row > 0:
                col = representation[:lowest_free_row, col_ix]
                cell_below = 1

                # Needed for hole_depth
                number_of_full_cells_above = np.sum(col)

                # There is at least one column_transition from the highest full cell to "the top".
                column_transitions += 1
                for row_ix, cell in enumerate(col):
                    if cell == 0:
                        # Holes
                        holes += 1
                        rows_with_holes_set.add(row_ix)
                        hole_depth += number_of_full_cells_above

                        # Column transitions
                        if cell_below:
                            column_transitions += 1

                        # Row transitions and wells
                        # Because col_ix == 0, all left_cells are 1
                        row_transitions += 1
                        cell_right = representation[row_ix, col_ix + 1]
                        if cell_right:
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0

                    else:  # cell is 1!
                        local_well_streak = 0

                        # Keep track of full cells above for hole_depth-feature
                        number_of_full_cells_above -= 1

                        # Column transitions
                        if not cell_below:
                            column_transitions += 1

                    # Define 'cell_below' for next (higher!) cell.
                    cell_below = cell

            # Check wells until lowest_free_row_right
            # Check transitions until lowest_free_row_left
            max_well_possibility = lowest_free_rows[col_ix + 1]
            if max_well_possibility > lowest_free_row:
                for row_ix in range(lowest_free_row, max_well_possibility):
                    cell_right = representation[row_ix, col_ix + 1]
                    if cell_right:
                        local_well_streak += 1
                        cumulative_wells += local_well_streak
                    else:
                        local_well_streak = 0
            # # Add row transitions for each empty cell above lowest_free_row
            row_transitions += (num_rows - lowest_free_row)

        elif col_ix == num_columns - 1:
            local_well_streak = 0
            if lowest_free_row > 0:
                col = representation[:lowest_free_row, col_ix]
                cell_below = 1

                # Needed for hole_depth
                number_of_full_cells_above = np.sum(col)

                # There is at least one column_transition from the highest full cell to "the top".
                column_transitions += 1
                for row_ix, cell in enumerate(col):
                    if cell == 0:
                        # Holes
                        holes += 1
                        rows_with_holes_set.add(row_ix)
                        hole_depth += number_of_full_cells_above

                        # Column transitions
                        if cell_below:
                            column_transitions += 1

                        # Wells and row transitions
                        # Because this is the last column (the right border is "full") and cell == 0:
                        row_transitions += 1
                        cell_left = representation[row_ix, col_ix - 1]
                        if cell_left:
                            row_transitions += 1
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0

                    else:  # cell is 1!
                        local_well_streak = 0

                        # Keep track of full cells above for hole_depth-feature
                        number_of_full_cells_above -= 1

                        # Column transitions
                        if not cell_below:
                            column_transitions += 1

                        # Row transitions
                        cell_left = representation[row_ix, col_ix - 1]
                        if not cell_left:
                            row_transitions += 1

                    # Define 'cell_below' for next (higher!) cell.
                    cell_below = cell

            # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
            # Check transitions until lowest_free_row_left
            max_well_possibility = lowest_free_rows[col_ix - 1]
            if max_well_possibility > lowest_free_row:
                for row_ix in range(lowest_free_row, max_well_possibility):
                    cell_left = representation[row_ix, col_ix - 1]
                    if cell_left:
                        row_transitions += 1
                        local_well_streak += 1
                        cumulative_wells += local_well_streak
                    else:
                        local_well_streak = 0
            # # Add row transitions from last column to border
            row_transitions += (num_rows - lowest_free_row)
        else:
            local_well_streak = 0
            if lowest_free_row > 0:
                col = representation[:lowest_free_row, col_ix]
                cell_below = 1

                # Needed for hole_depth
                number_of_full_cells_above = np.sum(col)

                # There is at least one column_transition from the highest full cell to "the top".
                column_transitions += 1
                for row_ix, cell in enumerate(col):
                    if cell == 0:
                        # Holes
                        holes += 1
                        rows_with_holes_set.add(row_ix)
                        hole_depth += number_of_full_cells_above

                        # Column transitions
                        if cell_below:
                            column_transitions += 1

                        # Wells and row transitions
                        cell_left = representation[row_ix, col_ix - 1]
                        cell_right = representation[row_ix, col_ix + 1]
                        if cell_left:
                            row_transitions += 1
                            if cell_right:
                                local_well_streak += 1
                                cumulative_wells += local_well_streak
                            else:
                                local_well_streak = 0
                        else:
                            local_well_streak = 0

                    else:  # cell is 1!
                        local_well_streak = 0
                        # Keep track of full cells above for hole_depth-feature
                        number_of_full_cells_above -= 1

                        # Column transitions
                        if not cell_below:
                            column_transitions += 1

                        # Row transitions
                        cell_left = representation[row_ix, col_ix - 1]
                        if not cell_left:
                            row_transitions += 1

                    # Define 'cell_below' for next (higher!) cell.
                    cell_below = cell
            # Check wells until minimum(lowest_free_row_left, lowest_free_row_right)
            # Check transitions until lowest_free_row_left
            lowest_free_row_left = lowest_free_rows[col_ix - 1]
            lowest_free_row_right = lowest_free_rows[col_ix + 1]
            max_well_possibility = np.minimum(lowest_free_row_left, lowest_free_row_right)
            if max_well_possibility > lowest_free_row:
                for row_ix in range(lowest_free_row, max_well_possibility):
                    cell_left = representation[row_ix, col_ix - 1]
                    cell_right = representation[row_ix, col_ix + 1]
                    if cell_left:
                        row_transitions += 1
                        if cell_right:
                            local_well_streak += 1
                            cumulative_wells += local_well_streak
                        else:
                            local_well_streak = 0
                    else:
                        local_well_streak = 0
            if lowest_free_row_left > max_well_possibility > lowest_free_row:
                for row_ix in range(max_well_possibility, lowest_free_row_left):
                    cell_left = representation[row_ix, col_ix - 1]
                    if cell_left:
                        row_transitions += 1

    rows_with_holes_set.remove(100)
    rows_with_holes = len(rows_with_holes_set)
    # if paper_order:
    out = [rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth]
    # else:  # ordered by standardized bcts-weights ['eroded', 'rows_with_holes', 'landing_height', 'column_transitions', 'holes', 'cumulative_wells', 'row_transitions', 'hole_depth']
    #     out = [rows_with_holes, column_transitions, holes, cumulative_wells, row_transitions, hole_depth]
    return out



