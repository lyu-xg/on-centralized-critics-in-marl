import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple
from collections import deque
from configs import (
    get_is_state_based_critic,
    get_normalization_reward,
    get_discount_factor,
    get_n_agents,
    is_reward_func_aligned,
    get_trace_len,
)
from utils.helpers import get_obs_space_dim

eps = np.finfo(np.float32).eps.item()


class TrajectoryBuffer:
    def __init__(self, env):
        self.env = env
        self.n_agents = get_n_agents()
        self.discount_factor = get_discount_factor()
        self.padding = np.zeros(
            (self.n_agents, get_obs_space_dim(self.env.observation_space))
        )
        self.record_state = get_is_state_based_critic()
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        if self.record_state:
            self.states = []

    def add_initial_state(
        self, initial_observation: np.ndarray, initial_state: np.ndarray = None
    ):
        if initial_state is not None:
            assert self.record_state
            assert len(self.states) == len(self.observations)
        if len(self.observations) == 0:
            self.observations = [initial_observation]
            if initial_state is not None:
                self.states = [initial_state]
        else:
            self.observations[-1] = initial_observation
            if initial_state is not None:
                self.states[-1] = initial_state

    def append(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        done: bool,
        state: np.ndarray = None,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if self.record_state:
            self.states.append(state)

    @staticmethod
    def get_discounted_return(rewards, dones) -> Tensor:
        R = 0.0
        gamma = get_discount_factor()
        returns = np.zeros(len(rewards))
        # if self.intrinsic_reward:
        #     rewards = self.intrinsic_reward.reward(rewards, self.get_states(), self.get_next_states(), self.actions)
        # is_terminal actually means that the next state is terminal state
        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = 0.0
            R = rewards[i] + gamma * R
            returns[i] = R
        if get_normalization_reward():
            returns = (returns - returns.mean()) / (returns.std() + eps)
        # return torch.tensor(returns).view((-1, 1))
        return returns.reshape((-1, 1))

    def get_traces(self):
        # return shape: [n_step, trace_len, n_agent, obs_len]
        trace_len = get_trace_len()
        hist = deque([self.padding] * trace_len)
        res = []
        for s, is_done in zip(self.observations, self.dones + [True]):
            hist.append(s)
            if len(hist) > trace_len:
                hist.popleft()
            res.append(torch.tensor(hist))
            if is_done:
                hist = deque([self.padding] * trace_len)
        return torch.stack(res)

    def __len__(self):
        return len(self.actions)


class JointTrajectoryBuffer(TrajectoryBuffer):
    """
        Joint: Designed for Multiple Agents
    """

    def __init__(self, env):
        super().__init__(env)
        self.aligned_reward = is_reward_func_aligned()

    def append(self, action, observation, reward, done, state):
        assert len(action) == len(observation) == self.n_agents
        assert type(done) is bool
        if self.aligned_reward:
            assert type(reward) is float
        else:
            assert (
                all(type(r) is np.float64 or type(r) is float for r in reward)
                and len(reward) is self.n_agents
            )
        super().append(action, observation, reward, done, state)

    def interpolate(self, trace):
        # replicate `trace` n_agent number of times
        return [[r for _ in range(self.n_agents)] for r in trace]

    def calc_return(self, joint_reward, culmulative):
        if self.aligned_reward:
            R = (
                self.get_discounted_return(self.rewards, self.dones)
                if culmulative
                else self.rewards
            )
            return R if joint_reward else self.interpolate(R)

        returns = np.array(self.rewards)
        if culmulative:
            returns = np.hstack(
                [
                    self.get_discounted_return(returns[:, a], self.dones)
                    for a in range(self.n_agents)
                ]
            )
        if joint_reward:
            returns = returns.mean(axis=1)
        return returns

    def get_trajectory(self, joint_reward=False):
        # used by central critic and get_individual_trajectory method
        return (
            self.get_traces(),  # [n_step, trace_len, n_agent, obs_len]
            self.actions,
            self.calc_return(joint_reward, culmulative=False),
            self.calc_return(joint_reward, culmulative=True),
            self.dones if joint_reward else self.interpolate(self.dones),
            torch.tensor(self.states).unsqueeze(1)
            if self.record_state
            else [0 for _ in self.observations],  # fake states
        )

    def get_individual_trajectory(self):
        # return trajectorys for reach agent
        traces, actions, rewards, returns, dones, states = self.get_trajectory(
            joint_reward=False
        )
        for i in range(self.n_agents):
            yield (
                torch.stack([h[:, i] for h in traces]),
                torch.tensor([a[i] for a in actions]),
                torch.tensor([r[i] for r in rewards]),
                torch.tensor([r[i] for r in returns]),
                [d[i] for d in dones],
                states,
            )
