import numpy as np
import torch
from torch.distributions import Categorical

from configs import (
    get_entropy_scale,
    get_critic_learning_ratio,
    get_is_centralized_critic,
    get_use_predefined_value_func,
)
from agents.reinforce import REINFORCE


class AC(REINFORCE):
    # we implement Q actor critic without baseline
    def learn(self, traces, actions, rewards, returns, dones, states) -> None:
        rewards = rewards.view(-1, 1)
        cen_critic = get_is_centralized_critic()

        action_logits, values = self.get_all_action_and_value(traces[:-1], states[:-1])
        action_dist = Categorical(logits=action_logits)

        action_logits_prime, values_prime = self.get_all_action_and_value(
            traces[1:], states[1:]
        )
        values_prime *= torch.tensor(1 - np.array(dones)).view(-1, 1)

        advantage = rewards + self.gamma * values_prime.detach() - values

        policy_loss = -1 * (action_dist.log_prob(actions) * advantage.detach()).mean()

        if cen_critic:
            value_loss = 0
        else:
            _, target_values_prime = self.get_all_action_and_value(
                traces[1:], states[1:], target_val=True
            )
            value_loss = (
                (rewards + self.gamma * target_values_prime.detach() - values)
                .pow(2)
                .mean()
            )

        # reset gradients
        self.optimizer.zero_grad()
        loss = (
            policy_loss
            + value_loss * get_critic_learning_ratio()
            - action_dist.entropy().mean() * get_entropy_scale()
        )
        # perform backprop
        loss.backward()
        self.optimizer.step()
