import numpy as np


class Checkers:
    def __init__(self) -> None:
        self.rows = 8
        self.cols = 8

    def __repr__(self):
        return "Checkers"

    def get_initial_state(self):
        board = np.zeros((self.rows, self.cols))
        for i in list(range(0, 3)) + list(range(5, 8)):
            for j in range((i + 1) % 2, 8, 2):
                board[i, j] = 1 if i < 4 else -1
        return board

    def get_next_state(self, state, action):
        ix, iy = action[0]
        jx, jy = action[1]
        state[jx, jy] = state[ix, iy]
        state[ix, iy] = 0
        return state

    def get_valid_moves(self, state):
        pass

    def check_win(self, state, action):
        pass

    def get_value_and_terminated(self, state, action):
        pass

    def get_opponent(self, player):
        pass

    def get_opponent_value(self, value):
        pass

    def change_perspective(self, state, player):
        pass

    def get_encoded_state(self, state):
        pass


game = Checkers()
state = game.get_initial_state()
print(state)
state = game.get_next_state(state, [[2, 1], [3, 0]])
print(state)
