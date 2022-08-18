import torch
import numpy as np
from torch import Tensor
from torch import nn
from configs import (
    get_critic_type,
    get_is_state_based_critic,
    get_network,
    get_is_q_func,
    get_is_dueling,
)
from envs import Converter
from modules.basics import SimpleLSTM, LinearWithTraces, Collect


class Model(nn.Module):
    def __init__(
        self,
        observation_converter: Converter,
        action_converter: Converter,
        state_converter: Converter,
    ):
        assert (
            len(observation_converter.shape) == 1
        ), "Only flat obs spaces supported by MLP model"
        assert (
            len(action_converter.shape) == 1
        ), "Only flat action spaces supported by MLP model"
        assert (
            len(state_converter.shape) == 1
        ), "Only flat state spaces supported by MLP model"
        super().__init__()
        self.observation_converter, self.action_converter, self.state_converter = (
            observation_converter,
            action_converter,
            state_converter,
        )

    def forward(self):
        raise NotImplementedError

    def get_features(self, input_dim):
        modules = []
        self.prev_out_dim = input_dim
        for layer in get_network():
            layer_type = SimpleLSTM if layer["recurrency"] else LinearWithTraces
            modules += [
                layer_type(self.prev_out_dim, layer["h_size"]),
                nn.Tanh(),
            ]
            self.prev_out_dim = layer["h_size"]

        return nn.Sequential(*modules, Collect())

    def get_actor_features(self):
        return self.get_features(input_dim=self.observation_converter.shape[0])

    def get_critic_features(self):
        if get_critic_type == 'state':
            input_dim = self.state_converter.shape[0]
        if get_critic_type == 'state+history':
            input_dim = self.state_converter.shape[0] + self.observation_converter.shape[0]
        if get_critic_type == 'history':
            input_dim = self.observation_converter.shape[0]

        return self.get_features(input_dim)

    def get_action_head(self):
        return self.action_converter.policy_out_model(in_features=self.prev_out_dim)

    def get_value_head(self):
        if get_is_dueling():
            return DuelingHead(self.prev_out_dim, self.action_converter)
        if get_is_q_func():
            return self.action_converter.policy_out_model(in_features=self.prev_out_dim)
        return nn.Linear(self.prev_out_dim, 1)

    @staticmethod
    def type_safe(x):
        if type(x) is list:
            x = np.array(x)
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        return x.float()


class DuelingHead(nn.Module):
    def __init__(self, prev_out_dim, action_converter):
        super().__init__()
        self.a = action_converter.policy_out_model(in_features=prev_out_dim)
        self.v = nn.Linear(prev_out_dim, 1)

    def forward(self, x):
        v = self.v(x)
        a = self.a(x)
        a_bar = a.mean()
        q = v + a - a_bar
        return v, a, q


class CombinedSeperateModel(Model):
    """
        A Full Actor Critic Model without shared weights
    """

    def __init__(self, o, a, s):
        super().__init__(o, a, s)
        self.actor = ActorModel(o, a, s)
        self.critic, self.target_critic = CriticModel(o, a, s), CriticModel(o, a, s)

    def forward(self, traces, states=None, action_only=False, target_val=False):
        a = self.actor(traces)
        if action_only:
            return a
        critic = self.target_critic if target_val else self.critic
        if get_critic_type() == 'state+history':
            input_data = torch.cat((states, traces), -1)
        else:
            input_data = states if get_critic_type() == 'state' else traces
        v = critic(input_data)
        return a, v

    def update_target_net(self):
        self.target_critic.load_state_dict(self.critic.state_dict())


class CombinedSharedModel(Model):
    """
        A Full Actor Critic Model with shared weights for the feature layers
    """

    def __init__(self, o, a, s):
        super().__init__(o, a, s)
        self.net = FeatureSharedModel(o, a, s)
        self.target_net = FeatureSharedModel(o, a, s)
        assert not get_is_state_based_critic()

    def forward(self, traces, states=None, action_only=False, target_val=False):
        net = self.target_net if target_val else self.net
        a, v = net(traces)
        return a if action_only else (a, v)

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())


class ActorModel(Model):
    def __init__(self, o: Converter, a: Converter, s: Converter):
        super().__init__(o, a, s)
        self.features = self.get_actor_features()
        self.action_head = self.get_action_head()

    def forward(self, x: Tensor) -> Tensor:
        return self.action_head(self.features(self.type_safe(x)))


class CriticModel(Model):
    def __init__(self, o: Converter, a: Converter, s: Converter):
        super().__init__(o, a, s)
        self.features = self.get_critic_features()
        self.value_head = self.get_value_head()

    def forward(self, x: Tensor) -> Tensor:
        return self.value_head(self.features(self.type_safe(x)))


class FeatureSharedModel(Model):
    def __init__(self, o: Converter, a: Converter, s: Converter):
        super().__init__(o, a, s)
        self.features = self.get_actor_features()
        self.value_head = self.get_value_head()
        self.action_head = self.get_action_head()

    def forward(self, x: Tensor) -> Tensor:
        return (
            self.value_head(self.features(self.type_safe(x))),
            self.action_head(self.type_safe(x)),
        )
