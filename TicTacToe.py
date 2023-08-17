import numpy as np


class TicTacToe:
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.action_size = self.rows * self.cols

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        return np.zeros((self.rows, self.cols))

    def get_next_state(self, state, action, player):
        row = action // self.rows
        col = action % self.cols
        state[row, col] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None:
            return False

        row = action // self.rows
        col = action % self.cols
        player = state[row, col]

        return (
            np.sum(state[row, :]) == player * self.cols or
            np.sum(state[:, col]) == player * self.rows or
            np.sum(np.diag(state)) == player * self.rows or
            np.sum(np.diag(np.flip(state, axis=0))) == player * self.rows
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
            (state == 1, state == 0, state == -1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
