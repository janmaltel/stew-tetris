import numpy as np
from tetris import state


class Tetromino:
    def __init__(self, feature_type, num_features, num_columns):
        self.feature_type = feature_type
        self.num_features = num_features
        self.num_columns = num_columns


class TetrominoSampler:
    def __init__(self, tetrominos):
        self.tetrominos = tetrominos
        self.current_batch = np.random.permutation(len(self.tetrominos))

    def next_tetromino(self):
        if len(self.current_batch) == 0:
            self.current_batch = np.random.permutation(len(self.tetrominos))
        tetromino = self.tetrominos[self.current_batch[0]]
        self.current_batch = np.delete(self.current_batch, 0)
        return tetromino


class TetrominoSamplerRandom:
    def __init__(self, tetrominos):
        self.tetrominos = tetrominos

    def next_tetromino(self):
        return np.random.choice(a=self.tetrominos, size=1)


class Straight(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)

    def __repr__(self):
        return '''
██ ██ ██ ██'''

    def get_after_states(self, current_state):
        after_states = []
        # Vertical placements
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows):
            anchor_row = free_pos
            new_representation = current_state.representation.copy()
            new_representation[anchor_row:(anchor_row + 4), col_ix] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] += 4
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 4),
                                    pieces_per_changed_row=np.array([1, 1, 1, 1]),
                                    landing_height_bonus=1.5, num_features=self.num_features,
                                    feature_type=self.feature_type)

            after_states.append(new_state)

        # Horizontal placements
        max_col_index = self.num_columns - 3
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 4)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row, col_ix:(col_ix + 4)] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix:(col_ix + 4)] = anchor_row + 1
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([4]),
                                    landing_height_bonus=0, num_features=self.num_features,
                                    feature_type=self.feature_type)

            after_states.append(new_state)
        return after_states


class Square(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)

    def __repr__(self):
        return '''
██ ██ 
██ ██'''

    def get_after_states(self, current_state):
        after_states = []
        # Horizontal placements
        max_col_index = self.num_columns - 1
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row:(anchor_row + 2), col_ix:(col_ix + 2)] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([2, 2]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)
        return after_states


class SnakeR(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)


    def __repr__(self):
        return '''
   ██ ██ 
██ ██'''

    def get_after_states(self, current_state):
        after_states = []
        # Horizontal placements
        max_col_index = self.num_columns - 2
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            anchor_row = np.maximum(np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)]), current_state.lowest_free_rows[col_ix + 2] - 1)
            new_representation = current_state.representation.copy()
            new_representation[anchor_row, col_ix:(col_ix + 2)] = 1
            new_representation[anchor_row + 1, (col_ix + 1):(col_ix + 3)] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 1
            new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([2]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

        # Vertical placements
        max_col_index = self.num_columns - 1
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, current_state.lowest_free_rows[col_ix + 1])
            new_representation = current_state.representation.copy()
            new_representation[(anchor_row + 1):(anchor_row + 3), col_ix] = 1
            new_representation[anchor_row:(anchor_row + 2), col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 3
            new_lowest_free_rows[col_ix + 1] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)
        return after_states


class SnakeL(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)

    def __repr__(self):
        return '''
██ ██ 
   ██ ██'''

    def get_after_states(self, current_state):
        after_states = []
        # Horizontal placements
        max_col_index = self.num_columns - 2
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, np.max(current_state.lowest_free_rows[(col_ix + 1):(col_ix + 3)]))
            new_representation = current_state.representation.copy()
            new_representation[anchor_row, (col_ix + 1):(col_ix + 3)] = 1
            new_representation[anchor_row + 1, col_ix:(col_ix + 2)] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 2
            new_lowest_free_rows[col_ix + 2] = anchor_row + 1
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([2]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

        # Vertical placements
        max_col_index = self.num_columns - 1
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 1)
            new_representation = current_state.representation.copy()
            new_representation[anchor_row:(anchor_row + 2), col_ix] = 1
            new_representation[(anchor_row + 1):(anchor_row + 3), col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 2
            new_lowest_free_rows[col_ix + 1] = anchor_row + 3
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)
        return after_states


class T(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)

    def __repr__(self):
        return """
   ██
██ ██ ██"""

    def get_after_states(self, current_state):
        after_states = []
        # Horizontal placements
        max_col_index = self.num_columns - 2
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            # upside-down T
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
            new_representation[anchor_row + 1, col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[[col_ix, col_ix + 2]] = anchor_row + 1
            new_lowest_free_rows[col_ix + 1] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([3]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

            # T
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix + 1], np.max(current_state.lowest_free_rows[[col_ix, col_ix + 2]]) - 1)
            new_representation = current_state.representation.copy()
            new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
            new_representation[anchor_row, col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 3]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)

            after_states.append(new_state)

        # Vertical placements.
        max_col_index = self.num_columns - 1
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            # Single cell on left
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 1, current_state.lowest_free_rows[col_ix + 1])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row + 1, col_ix] = 1
            new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 2
            new_lowest_free_rows[col_ix + 1] = anchor_row + 3
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

            # Single cell on right
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 1)
            new_representation = current_state.representation.copy()
            new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
            new_representation[anchor_row + 1, col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 3
            new_lowest_free_rows[col_ix + 1] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)
        return after_states


class RCorner(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)

    def __repr__(self):
        return """
██ ██ ██
██"""

    def get_after_states(self, current_state):
        after_states = []
        # Horizontal placements
        max_col_index = self.num_columns - 2
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            # Bottom-right corner
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
            new_representation[anchor_row + 1, col_ix + 2] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix:(col_ix + 2)] = anchor_row + 1
            new_lowest_free_rows[col_ix + 2] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([3]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

            # Top-left corner
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], np.max(current_state.lowest_free_rows[(col_ix + 1):(col_ix + 3)]) - 1)
            new_representation = current_state.representation.copy()
            new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
            new_representation[anchor_row, col_ix] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 3]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

        # Vertical placements.
        max_col_index = self.num_columns - 1
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            # Top-right corner
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix] - 2, current_state.lowest_free_rows[col_ix + 1])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row + 2, col_ix] = 1
            new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 3),
                                    pieces_per_changed_row=np.array([1, 1, 2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

            # Bottom-left corner
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
            new_representation[anchor_row, col_ix + 1] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 3
            new_lowest_free_rows[col_ix + 1] = anchor_row + 1
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)
        return after_states


class LCorner(Tetromino):
    def __init__(self, feature_type, num_features, num_columns):
        Tetromino.__init__(self, feature_type, num_features, num_columns)

    def __repr__(self):
        return """
██ ██ ██
      ██"""

    def get_after_states(self, current_state):
        after_states = []
        max_col_index = self.num_columns - 2
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            # Bottom-left corner (= 'hole' in top-right corner)
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 3)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row, col_ix:(col_ix + 3)] = 1
            new_representation[anchor_row + 1, col_ix] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix] = anchor_row + 2
            new_lowest_free_rows[(col_ix + 1):(col_ix + 3)] = anchor_row + 1
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([3]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

            # Top-right corner
            anchor_row = np.maximum(np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)]) - 1, current_state.lowest_free_rows[col_ix + 2])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row + 1, col_ix:(col_ix + 3)] = 1
            new_representation[anchor_row, col_ix + 2] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix: (col_ix + 3)] = anchor_row + 2
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 2),
                                    pieces_per_changed_row=np.array([1, 3]),
                                    landing_height_bonus=0.5, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

        # Vertical placements. 'height' becomes 'width' :)
        max_col_index = self.num_columns - 1
        for col_ix, free_pos in enumerate(current_state.lowest_free_rows[:max_col_index]):
            # Top-left corner
            anchor_row = np.maximum(current_state.lowest_free_rows[col_ix], current_state.lowest_free_rows[col_ix + 1] - 2)
            new_representation = current_state.representation.copy()
            new_representation[anchor_row + 2, col_ix + 1] = 1
            new_representation[anchor_row:(anchor_row + 3), col_ix] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix: (col_ix + 2)] = anchor_row + 3
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 3),
                                    pieces_per_changed_row=np.array([1, 1, 2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)

            # Bottom-right corner
            anchor_row = np.max(current_state.lowest_free_rows[col_ix:(col_ix + 2)])
            new_representation = current_state.representation.copy()
            new_representation[anchor_row:(anchor_row + 3), col_ix + 1] = 1
            new_representation[anchor_row, col_ix] = 1
            new_lowest_free_rows = current_state.lowest_free_rows.copy()
            new_lowest_free_rows[col_ix + 1] = anchor_row + 3
            new_lowest_free_rows[col_ix] = anchor_row + 1
            new_state = state.State(representation=new_representation, lowest_free_rows=new_lowest_free_rows,
                                    anchor_col=col_ix,
                                    changed_lines=np.arange(anchor_row, anchor_row + 1),
                                    pieces_per_changed_row=np.array([2]),
                                    landing_height_bonus=1, num_features=self.num_features,
                                    feature_type=self.feature_type)
            after_states.append(new_state)
        return after_states
