import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm.notebook import trange

from MCTS import MCTS


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            # action_probs = action_probs ** (1 / self.args["temperature"])
            # action_probs /= np.sum(action_probs)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)

            value, ter = self.game.get_value_and_terminated(state, action)

            if ter:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if player == hist_player else self.game.get_opponent_value(
                        player)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))

                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:min(
                batchIdx + self.args["batch_size"], len(memory) - 1)]
            state, policy_target, value_target = zip(*sample)
            state, policy_target, value_target = np.array(state), np.array(
                policy_target), np.array(value_target).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.model.device)
            policy_target = torch.tensor(
                policy_target, dtype=torch.float32, device=self.model.device)
            value_target = torch.tensor(
                value_target, dtype=torch.float32, device=self.model.device)

            policy_out, value_out = self.model(state)
            policy_loss = F.cross_entropy(policy_out, policy_target)
            value_loss = F.mse_loss(value_out, value_target)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args["num_selfPlay_iterations"]):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args["num_epochs"]):
                self.train(memory)

            name = f"[{self.model.p1}+{self.model.p2}+{self.args['num_searches']}+{self.args['num_selfPlay_iterations']}]@{iteration}"

            torch.save(self.model.state_dict(),
                       f"./{self.game}/models/{name}.pt")
            torch.save(self.optimizer.state_dict(),
                       f"./{self.game}/optimizers/{name}.pt")
