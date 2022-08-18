import numpy as np
import torch
from torch.distributions import Categorical

from configs import (
    get_entropy_scale,
    get_critic_learning_ratio,
    get_discount_factor,
    get_epochs,
    get_batch_size,
    get_clip_grad_norm,
)
from agents.reinforce import REINFORCE


class A2C(REINFORCE):
    def __init__(self, observation_space, action_space, central_critic) -> None:
        super().__init__(observation_space, action_space, central_critic)
        self.gamma = get_discount_factor()

    def learn(self, states, actions, next_states, rewards, dones) -> None:
        """
        Training procedure. Calculates actor and critic loss and performs backprop.
        """
        rewards = torch.tensor(rewards).view(-1, 1)
        states = np.vstack(states)

        action_logits, values = self.get_all_action_and_value(states)
        _, next_value = self.get_all_action_and_value(next_states)

        # V = values.detach().view(-1).numpy()
        # R = torch.tensor(TrajectoryBuffer.get_discounted_return(V, dones))
        advantage = np.array(rewards).squeeze() - values.squeeze().detach().numpy()
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # dataset = NonSequentialSingleDataset(states, actions, next_states, advantage, rewards)
        # loader = DataLoader(dataset, batch_size=get_batch_size(), shuffle=True)

        # with torch.autograd.detect_anomaly():
        # for _ in range(get_epochs()):
        #     for states, actions, next_states, advantage, rewards in loader:

        if self.central_critic:
            policy = self.action_logits(states)
            value_loss = 0
        else:
            policy, value = self.get_action_and_value(states)
            _, next_value = self.get_action_and_value(next_states)
            value_loss = (next_value * self.gamma + rewards - value).pow(2).mean()

        action_dist = Categorical(logits=policy)
        action_log_prob = action_dist.log_prob(torch.tensor(actions))

        policy_loss = -1 * (action_log_prob * torch.tensor(advantage)).mean()

        # action_dist = Categorical(logits=action_logits)
        # adv = (R - values.detach()).view(-1)
        # policy_loss = (-1 * action_dist.log_prob(torch.tensor(actions)) * adv).mean()
        # value_loss = (R - values).pow(2).mean()

        # reset gradients
        self.optimizer.zero_grad()
        loss = (
            policy_loss
            + value_loss * get_critic_learning_ratio()
            - action_dist.entropy().mean() * get_entropy_scale()
        )
        # perform backprop
        # torch.nn.utils.clip_grad_norm_(self.parameters(), get_clip_grad_norm())
        loss.backward()
        self.optimizer.step()
