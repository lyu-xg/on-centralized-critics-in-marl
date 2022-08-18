import numpy as np
import wandb
import torch.optim as optim
from gym import Space
from torch.distributions import Categorical

from envs import Converter
from configs import (
    get_entropy_scale,
    get_learning_rate,
    get_common_feature,
    get_discount_factor,
    get_critic_learning_ratio,
)
from modules import (
    ActorModel,
    CentralCritic,
    CombinedSeperateModel,
    CombinedSharedModel,
)


class REINFORCE:
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        state_space: Space,
        central_critic: CentralCritic,
    ) -> None:
        self.observation_converter = Converter.for_space(observation_space)
        self.action_converter = Converter.for_space(action_space)
        self.state_converter = Converter.for_space(state_space)
        self.gamma = get_discount_factor()
        self.central_critic = central_critic

        if central_critic:
            self.actor_model = ActorModel(
                self.observation_converter, self.action_converter, self.state_converter
            )
            wandb.watch(self.actor_model)
        else:
            combined_model = (
                CombinedSharedModel if get_common_feature() else CombinedSeperateModel
            )
            self.combined_model = combined_model(
                self.observation_converter, self.action_converter, self.state_converter
            )
            wandb.watch(self.combined_model)

        self.optimizer = optim.Adam(self.parameters(), lr=get_learning_rate())

    def set_id(self, id):
        self._id = id

    def target_update(self):
        if not self.central_critic:
            self.combined_model.update_target_net()

    def action_logits(self, traces):
        if self.central_critic:
            return self.actor_model(traces)
        return self.combined_model(traces, action_only=True)

    def action_probs(self, traces):
        return Categorical(logits=self.action_logits(traces)).probs

    def get_all_action_and_value(self, traces, states, target_val=False):
        # called during training time
        if not self.central_critic:
            return self.combined_model(traces, states, target_val=target_val)
        # if we have central critic, let central critic figure all of this out
        return (
            self.action_logits(traces),
            self.central_critic.from_buffer(
                target_val=target_val, agent_id=self._id
            ).detach(),
        )

    def get_action_and_value(self, traces, states):
        assert not self.central_critic
        return self.combined_model(traces, states)
        # return self.action_logits(states), self.central_critic(states).detach()

    def parameters(self):
        if self.central_critic:
            return self.actor_model.parameters()
        return self.combined_model.parameters()

    def rand_action(self):
        a_dim = self.action_converter.shape[0]
        return np.random.randint(a_dim)

    def select_action(self, traces: np.ndarray) -> int:
        """
        Control procedure. Sample an action based on policy by performing forward pass.
        """
        # and sample an action using the distribution
        action = self.action_converter.action(self.action_logits(traces))
        return action.item()

    def learn(self, traces, actions, rewards, returns, dones, states) -> None:
        """
        Training procedure. Calculates actor and critic loss and performs backprop.
        """
        action_logits, values = self.get_all_action_and_value(traces[:-1], states[:-1])

        advantage = returns - values
        advantage.detach_()
        # advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + self.eps)

        action_dist = Categorical(logits=action_logits)

        policy_loss = (-1 * action_dist.log_prob(actions) * advantage.view(-1)).mean()

        value_loss = get_critic_learning_ratio() * (returns - values).pow(2).mean()

        # reset gradients
        self.optimizer.zero_grad()

        loss = (
            policy_loss
            + value_loss
            - action_dist.entropy().mean() * get_entropy_scale()
        )
        # perform backprop
        loss.backward()
        self.optimizer.step()
