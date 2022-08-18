import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from itertools import chain
from agents import REINFORCE
from configs import (
    get_use_both_value_loss,
    get_ppo_params,
    ex,
    get_entropy_scale,
    get_critic_learning_ratio,
    get_clip_grad_norm,
    get_icm_flag,
    get_learning_rate,
)
from utils.datasets import NonSequentialSingleDataset
from curiosity import ICM, NoCuriosity


class DPPO(REINFORCE):
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

        rewards = rewards.view(-1, 1)

        policy_old, (_, _, q_old) = self.get_all_action_and_value(
            traces[:-2], states[:-2]
        )
        a_log_prob_old = Categorical(logits=policy_old).log_prob(actions[:-1])

        advantage = (
            np.array(returns[:-1]).squeeze()
            - q_old.gather(1, actions[:-1].unsqueeze(1)).squeeze().detach().numpy()
        )

        # 舍掉最后一个transition
        dataset = NonSequentialSingleDataset(
            states[:-2],
            states[1:-1],
            traces[:-2],
            traces[1:-1],
            actions[:-1],
            actions[1:],
            advantage,
            rewards[:-1],
            returns[:-1],
            a_log_prob_old.detach(),
            dones[:-1],
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for (
                states,
                next_states,
                traces,
                next_traces,
                actions,
                next_actions,
                advantage,
                rewards,
                returns,
                action_log_prob_old,
                done,
            ) in loader:
                if self.central_critic:
                    policy = self.action_logits(traces)
                    value_loss = 0
                else:
                    policy, (v, a, q) = self.get_action_and_value(traces, states)
                    _, (_, _, q_prime) = self.get_action_and_value(
                        next_traces, next_states
                    )
                    # if get_ppo_value_clip():
                    #     v_clipped = value_old + (value - value_old).clamp(-self.clip_param, self.clip_param)
                    #     vloss1 = (value - returns).pow(2)
                    #     vloss2 = (v_clipped - returns).pow(2)
                    #     value_loss = torch.max(vloss1, vloss2).mean()

                    # q_loss = (rewards + self.gamma * q_prime.detach().gather(1, next_actions.unsqueeze(1))
                    #   - q.gather(1, actions.unsqueeze(1))).pow(2).mean()
                    q_loss = (returns - q.gather(1, actions.unsqueeze(1))).pow(2).mean()

                    v_loss = (
                        (v - returns).pow(2).mean() if get_use_both_value_loss() else 0
                    )
                    value_loss = q_loss + v_loss

                action_dist = Categorical(logits=policy)
                action_log_prob = action_dist.log_prob(actions)

                # action_advantage = advantage.gather(1, actions.unsqueeze(1))
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
