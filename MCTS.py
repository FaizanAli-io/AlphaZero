import math
import torch
import numpy as np


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visits=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.value = 0
        self.visits = visits
        self.children = []

    def is_fully_expanded(self):
        return len(self.children) > 0

    def get_ucb(self, child):
        if child.visits == 0:
            Q = 0
        else:
            Q = 1 - ((child.value / child.visits) + 1) / 2

        return Q + self.args["C"] * math.sqrt(self.visits / (child.visits + 1)) * child.prior

    def select(self):
        best_child = None
        best_score = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(
                    child_state, player=-1)
                child = Node(self.game, self.args,
                             child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value += value
        self.visits += 1

        if self.parent is not None:
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visits=1)

        # policy, _ = self.model(
        #     torch.tensor(self.game.get_encoded_state(state),
        #                  device=self.model.device).unsqueeze(0)
        # )

        # policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        # policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args["dirichlet_epsilon"] \
        #     * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)

        # valid_moves = self.game.get_valid_moves(state)
        # policy *= valid_moves
        # policy /= np.sum(policy)

        # root.expand(policy)

        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, ter = self.game.get_value_and_terminated(
                node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not ter:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(
                        node.state), device=self.model.device).unsqueeze(0)
                )

                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visits
        action_probs /= np.sum(action_probs)

        return action_probs
