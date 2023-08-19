import os
import random
import numpy as np

import torch
import torch.nn.functional as F

from MCTSParallel import MCTS

UP = '\033[1A'
CLEAR = '\x1b[2K'


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        spgs = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]
        return_memory = []
        player = 1

        while len(spgs) > 0:
            states = np.stack([spg.state for spg in spgs])

            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spgs)

            for i in range(len(spgs))[::-1]:
                spg = spgs[i]

                # Action probs based on visit count
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visits
                action_probs /= np.sum(action_probs)

                spg.memory.append((
                    spg.root.state,
                    action_probs,
                    player,
                ))

                # Apply temperature variance
                temperature_action_probs = action_probs ** \
                    (1 / self.args["temperature"])
                temperature_action_probs /= temperature_action_probs.sum()

                action = np.random.choice(
                    self.game.action_size, p=temperature_action_probs)

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, ter = self.game.get_value_and_terminated(
                    spg.state, action)

                if ter:
                    for (
                        hist_neutral_state,
                        hist_action_probs,
                        hist_player
                    ) in spg.memory:

                        hist_outcome = value if player == hist_player \
                            else self.game.get_opponent_value(value)

                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))

                    del spgs[i]
                    print(len(spgs), end=", ")

            player = self.game.get_opponent(player)

        print()
        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            print(batchIdx, end=", ")
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
        print()

    def learn(self):
        self.check_save_directory()

        for iteration in range(self.args["num_iterations"]):
            print(f"Epoch: {iteration + 1}")
            memory = []

            self.model.eval()
            for selfPlay_iterations in range(
                    self.args["num_selfPlay_iterations"] //
                    self.args["num_parallel_games"]):
                print(f"\tSelf Play Iteration: {selfPlay_iterations}")
                print("\t\tGames Left", end=": ")
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args["num_epochs"]):
                print(f"\tTraining Iteration: {epoch}")
                print("\t\tBatch Index", end=": ")
                self.train(memory)

            name = f"[{self.model.p1},{self.model.p2},"
            name += f"{self.args['num_searches']},"
            name += f"{self.args['num_selfPlay_iterations']}]"
            name += f"@{iteration}"

            torch.save(self.model.state_dict(),
                       f"./{self.game}/models/{name}.pt")
            torch.save(self.optimizer.state_dict(),
                       f"./{self.game}/optimizers/{name}.pt")

    def check_save_directory(self):
        if os.path.isdir(f"{self.game}"):
            if not os.path.isdir(f"{self.game}/models"):
                os.mkdir(f"{self.game}/models")
            if not os.path.isdir(f"{self.game}/optimizers"):
                os.mkdir(f"{self.game}/optimizers")
        else:
            os.mkdir(f"{self.game}")
            os.mkdir(f"{self.game}/models")
            os.mkdir(f"{self.game}/optimizers")


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
