import numpy as np


class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.action_size = self.cols
        self.consec = 4

    def __repr__(self):
        return "ConnectFour"

    def get_initial_state(self):
        return np.zeros((self.rows, self.cols))

    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.consec):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 or c < 0 or
                    r >= self.rows or
                    c >= self.cols or
                    state[r][c] != player
                ):
                    return i - 1
            return self.consec - 1

        return (
            count(1, 0) >= self.consec - 1  # vertical
            or (count(0, 1) + count(0, -1)) >= self.consec - 1  # horizontal
            # top left diagonal
            or (count(1, 1) + count(-1, -1)) >= self.consec - 1
            # top right diagonal
            or (count(1, -1) + count(-1, 1)) >= self.consec - 1
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        return encoded_state
