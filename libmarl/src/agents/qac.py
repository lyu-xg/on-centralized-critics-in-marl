import numpy as np
import torch
from gym import Space
from torch.distributions import Categorical

from envs import Converter
from envs.matrix_game import GuessColor, TrickyStagHunt
from configs import (
    get_env,
    get_entropy_scale,
    get_critic_learning_ratio,
    get_discount_factor,
    get_is_centralized_critic,
    get_use_predefined_value_func,
)
from agents.reinforce import REINFORCE


class QAC(REINFORCE):
    # we implement Q actor critic without baseline
    def learn(self, traces, actions, rewards, returns, dones, states) -> None:
        # rewards = torch.tensor(rewards).view(-1, 1)
        # actions = torch.tensor(actions)
        rewards = rewards.view(-1, 1)
        cen_critic = get_is_centralized_critic()

        action_logits, values = self.get_all_action_and_value(traces[:-1], states[:-1])
        action_dist = Categorical(logits=action_logits)

        if get_use_predefined_value_func() and not cen_critic:
            # for decentralized critic that uses predifined value func
            raise NotImplementedError  # no longer supported due to advantage change
            update_values = torch.tensor(
                [
                    TrickyStagHunt.true_decen_q(s, a)
                    for s, a in zip(traces[:-1, -1], actions)
                ]
            )
        elif not cen_critic:
            raise not NotImplementedError  # not supporting this due to advantage change
            update_values = values.gather(1, actions.unsqueeze(1))
        else:  # central critic already did the gathering for us
            # update_values = values
            update_values = self.central_critic.from_buffer(
                agent_id=self._id, advantage=True
            )

        policy_loss = (
            -1 * (action_dist.log_prob(actions) * update_values.detach()).mean()
        )

        if cen_critic:
            value_loss = 0
        else:
            _, value_prime = self.get_all_action_and_value(traces[1:], states[1:])
            # value_prime = value_prime.max(dim=1).values.unsqueeze(1) # v value of state_prime
            value_prime = value_prime[:-1].gather(1, actions[1:].unsqueeze(1)).detach()
            value_prime *= torch.tensor(
                1 - np.array(dones)[1].reshape((-1, 1)), dtype=int
            )
            value_loss = (
                (rewards[:-1] + self.gamma * value_prime - update_values[:-1])
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
