import torch
import wandb
from torch.distributions import Categorical
from typing import List
from utils import TrajectoryBuffer
from agents import PPO, AC, QAC, Qnet, REINFORCE, A2C, DPPO
from modules import CentralCritic
from envs import Env
from configs import (
    get_copy_target_net_interval,
    get_method,
    get_n_agents,
    get_is_centralized_critic,
    get_rand_walk,
    get_is_share_param,
)


class Team:
    def __init__(self, exp_buffer: TrajectoryBuffer, env: Env):
        self.method = get_method()
        self.exp_buffer = exp_buffer
        self.env = env
        self.central_critic = (
            self.instantiate_central_critic() if get_is_centralized_critic() else None
        )

        if get_is_share_param:
            agent = self.instantiate_agent()
            self.agents = [agent for _ in range(get_n_agents())]
        else:
            self.agents = [self.instantiate_agent() for _ in range(get_n_agents())]
        self.set_agent_ids()
        self.add_agents_to_central_critic()

        self.copy_target_net_interval = get_copy_target_net_interval()
        self.n_learn = 0

    def act(self, traces):
        if get_rand_walk():
            return [agent.rand_action() for agent in self.agents]
        return [
            a.select_action(torch.tensor(traces)[:, i].unsqueeze(0))
            for i, a in enumerate(self.agents)
        ]

    def learn(self):
        if get_rand_walk():
            return

        for agent, (traces, action, reward, returns, done, states) in zip(
            self.agents, self.exp_buffer.get_individual_trajectory()
        ):
            agent.learn(traces, action, reward, returns, done, states)
        if self.central_critic:
            self.central_critic.learn()
        self.target_net_update()

    def target_net_update(self):
        self.n_learn += 1
        if self.n_learn % self.copy_target_net_interval == 0:
            for a in self.agents:
                a.target_update()
            if self.central_critic:
                self.central_critic.update_target()

    def instantiate_agent(self):
        agent = {
            "ppo": PPO,
            "dppo": DPPO,
            "ac": AC,
            "qac": QAC,
            "reinforce": REINFORCE,
            "qnet": Qnet,
            "a2c": A2C,
        }[self.method]

        return agent(
            self.env.observation_space,
            self.env.action_space,
            self.env.state_space,
            self.central_critic,
        )

    def instantiate_central_critic(self):
        central_critic = CentralCritic(self.env, self.exp_buffer)
        wandb.watch(central_critic)
        return central_critic

    def add_agents_to_central_critic(self):
        if self.central_critic:
            self.central_critic.add_agents(self.agents)

    def set_agent_ids(self):
        for i, a in enumerate(self.agents):
            a.set_id(i)

    def get_action_probs(self, traces):
        # to be used for telemetry
        return [
            Categorical(logits=agent.action_logits(traces)).probs.detach()
            for agent in self.agents
        ]

    def get_values(self, traces, states):
        # to be used for telemetry
        return [
            v for _, v in (a.get_action_and_value(traces, states) for a in self.agents)
        ]
