import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from itertools import chain
from agents import REINFORCE
from configs import (
    get_ppo_value_clip,
    get_ppo_params,
    get_entropy_scale,
    get_critic_learning_ratio,
    get_clip_grad_norm,
    get_icm_flag,
    get_learning_rate,
)
from utils.datasets import NonSequentialSingleDataset
from curiosity import ICM, NoCuriosity


class PPO(REINFORCE):
    def __init__(self, o_conv, a_conv, s_conv, cen_critic):
        super().__init__(o_conv, a_conv, s_conv, cen_critic)
        (
            self.buffer_size,
            self.batch_size,
            self.epochs,
            self.clip_param,
        ) = get_ppo_params()

        curiosity = ICM if get_icm_flag() else NoCuriosity
        self.curiosity = curiosity(self.state_converter, self.action_converter)

        self.optimizer = optim.Adam(
            chain(self.parameters(), self.curiosity.parameters()),
            lr=get_learning_rate(),
        )

    def learn(self, traces, actions, rewards, returns, dones, states):
        # the crux of PPO, try to learn more batches from each rollout for sample efficiency
        # but not updating policy too far off the original policy for stability

        returns = self.curiosity.reward(returns)

        policy_old, values_old = self.get_all_action_and_value(traces[:-1], states[:-1])
        a_log_prob_old = Categorical(logits=policy_old).log_prob(actions)

        # advantage = np.array(returns).squeeze() - values_old.squeeze().detach().numpy()

        _, next_values = self.get_all_action_and_value(traces[1:], states[1:])
        advantage = next_values.squeeze().detach() - values_old.squeeze().detach()

        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        dataset = NonSequentialSingleDataset(
            states[:-1],
            traces[:-1],
            traces[1:],
            actions,
            advantage,
            returns,
            a_log_prob_old.detach(),
            values_old.detach(),
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # with torch.autograd.detect_anomaly():
        for _ in range(self.epochs):
            for (
                states,
                traces,
                next_traces,
                actions,
                advantage,
                returns,
                action_log_prob_old,
                value_old,
            ) in loader:
                if self.central_critic:
                    policy = self.action_logits(traces)
                    value_loss = 0
                else:
                    policy, value = self.get_action_and_value(traces, states)
                    if get_ppo_value_clip():
                        v_clipped = value_old + (value - value_old).clamp(
                            -self.clip_param, self.clip_param
                        )
                        vloss1 = (value - returns).pow(2)
                        vloss2 = (v_clipped - returns).pow(2)
                        value_loss = torch.max(vloss1, vloss2).mean()
                    else:
                        value_loss = (value - returns).pow(2).mean()

                action_dist = Categorical(logits=policy)
                action_log_prob = action_dist.log_prob(actions)

                ratio = torch.exp(action_log_prob + 1e-8) / (
                    torch.exp(action_log_prob_old) + 1e-8
                )
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * advantage
                )
                policy_loss = -1 * torch.min(surr1, surr2).mean()

                entropy = action_dist.entropy().mean()

                self.optimizer.zero_grad()
                loss = (
                    value_loss * get_critic_learning_ratio()
                    - get_entropy_scale() * entropy
                    + policy_loss
                )
                loss = self.curiosity.loss(loss, traces, next_traces, actions)
                # ex.log_scalar('total_loss', loss.item())
                # wandb.log({'total_loss': loss.item()})
                torch.nn.utils.clip_grad_norm_(self.parameters(), get_clip_grad_norm())
                loss.backward()
                self.optimizer.step()
