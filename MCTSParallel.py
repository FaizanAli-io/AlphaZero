import math
import torch
import numpy as np


class Node:
    def __init__(self, game, args, state,
                 parent=None, action_taken=None,
                 prior=0, visits=0):
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

        return self.args["C"] * math.sqrt(self.visits / (child.visits + 1)) * \
            child.prior + Q

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
    def search(self, states, spgs):
        # Get Policy from Model
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(states),
                device=self.model.device,
            )
        )

        # Apply Dirichlet Distribution to Policy
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + \
            self.args["dirichlet_epsilon"] * \
            np.random.dirichlet(
                [self.args["dirichlet_alpha"]] * self.game.action_size,
                size=policy.shape[0],
        )

        # Get masked policy via valid moves
        for i, spg in enumerate(spgs):
            spg_policy = policy[i]
            spg_policy *= self.game.get_valid_moves(states[i])
            spg_policy /= np.sum(spg_policy)
            spg.root = Node(self.game, self.args, states[i], visits=1)
            spg.root.expand(spg_policy)

        # Run required searches for each root
        for search in range(self.args["num_searches"]):
            for spg in spgs:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, ter = self.game.get_value_and_terminated(
                    node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if ter:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandableGames = [mappingIdx for mappingIdx in range(len(spgs))
                               if spgs[mappingIdx].node is not None]

            if len(expandableGames) > 0:
                states = np.stack([spgs[mappingIdx].node.state for mappingIdx
                                   in expandableGames])

                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(states),
                        device=self.model.device,
                    )
                )

                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

                for i, mappingIdx in enumerate(expandableGames):
                    node = spgs[mappingIdx].node
                    spg_policy, spg_value = policy[i], value[i]

                    spg_policy *= self.game.get_valid_moves(node.state)
                    spg_policy /= np.sum(spg_policy)
                    node.expand(spg_policy)
                    node.backpropagate(spg_value)
