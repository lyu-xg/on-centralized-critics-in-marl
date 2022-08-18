# NOT WORKING
import numpy as np
from collections import namedtuple
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from gym import Space
from torch import Tensor

from envs import Converter
from configs import get_ppo_params, get_learning_rate, get_discount_factor
from utils.datasets import NonSequentialSingleTDDataset

SavedAction = namedtuple(
    "SavedAction", ["action_dist", "action_selected", "state", "value"]
)
Transition = namedtuple(
    "Transition",
    ["state", "action", "action_dist", "state_value", "reward", "is_terminal"],
)


EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay


class Model(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, state_converter: Converter, action_converter: Converter):
        assert (
            len(state_converter.shape) == 1
        ), "Only flat spaces supported by MLP model"
        assert (
            len(action_converter.shape) == 1
        ), "Only flat action spaces supported by MLP model"
        super(Model, self).__init__()
        (
            self.buffer_size,
            self.batch_size,
            self.epochs,
            self.clip_param,
        ) = get_ppo_params()
        h_size = 64

        self.features = nn.Sequential(
            nn.Linear(state_converter.shape[0], h_size), nn.ReLU(),
        )
        self.value_head = nn.Linear(h_size, 2)  # TODO hardcoded a_size

    def forward(self, x: Tensor) -> Tensor:
        return self.value_head(self.features(x.float()))

    def qval(self, states: np.ndarray, actions: np.ndarray) -> Tensor:
        out = self.forward(states)
        a = actions.view(-1, 1)
        return torch.gather(out, 1, a).view(-1)


class Qnet:
    def __init__(self, observation_space: Space, action_space: Space) -> None:
        self.gamma = get_discount_factor()
        (
            self.buffer_size,
            self.batch_size,
            self.epochs,
            self.clip_param,
        ) = get_ppo_params()
        self.buffer = []
        self.state_converter = Converter.for_space(observation_space)
        self.action_converter = Converter.for_space(action_space)
        self.model = Model(self.state_converter, self.action_converter)
        self.optimizer = optim.Adam(self.model.parameters(), lr=get_learning_rate())
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Control code. Sample an action based on policy by performing forward pass.
        """
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        sample = random.random()

        if sample < eps_threshold:
            action = random.randint(0, 1)
        else:
            q = self.model(torch.from_numpy(state).float())
            action = q.argmax().item()
        self.last_action = SavedAction(None, action, state, None)
        return action

    def append_transition(self, reward: int, is_terminal: bool) -> None:
        # self.rewards.append(reward)
        self.steps_done += 1
        action_probs, action, state, state_value = self.last_action
        self.buffer.append(
            Transition(state, action, action_probs, state_value, reward, is_terminal)
        )

    def learn(self) -> None:
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """

        (states, actions, _, _, rewards, dones) = zip(*self.buffer)

        dataset = NonSequentialSingleTDDataset(states, actions, rewards, dones)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with torch.autograd.detect_anomaly():
            for _ in range(self.epochs):
                for s, a, r, t, s1 in loader:
                    q = self.model.qval(s, a)
                    q1 = self.model(s1).max(dim=1).values.detach().float()
                    loss = F.smooth_l1_loss(
                        q.float(), (r - t.int()).float() + self.gamma * q1
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
