import torch
import numpy as np
from torch import nn, Tensor
from configs import (
    get_n_agents,
    get_discount_factor,
    get_network,
    get_is_q_func,
    get_use_predefined_value_func,
    get_critic_type,
    get_trace_len,
    get_is_state_based_critic,
    get_is_history_based_critic,
    get_learning_rate_for_critic,
    get_use_marginalized_q,
    get_is_q_func,
)
from modules.basics import SimpleLSTM, LinearWithTraces, Collect
from envs import Converter
from utils import TrajectoryBuffer
from envs.matrix_game import TrickyStagHunt


class JointCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_converter = Converter.for_space(env.observation_space)
        action_converter = Converter.for_space(env.action_space)
        state_converter = Converter.for_space(env.state_space)

        state_dim = state_converter.shape[0]
        obs_dim = obs_converter.shape[0] * get_n_agents()
        
        input_dim = 0
        if get_is_state_based_critic():
            input_dim += state_dim
        if get_is_history_based_critic():
            input_dim += obs_dim

        joint_action_space_dim = action_converter.shape[0] ** get_n_agents()

        modules = []
        for layer in get_network():
            layer_module = SimpleLSTM if layer["recurrency"] else LinearWithTraces
            modules += [layer_module(input_dim, layer["h_size"]), nn.Tanh()]
            input_dim = layer["h_size"]
        modules.append(Collect())
        if get_is_q_func():
            modules.append(
                action_converter.policy_out_model(input_dim, joint_action_space_dim)
            )
        else:
            modules.append(nn.Linear(input_dim, 1))

        self.value = nn.Sequential(*modules)

    def forward(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        return self.value(x.float())


class CentralCritic(nn.Module):
    def __init__(self, env, buffer: TrajectoryBuffer):
        super().__init__()
        self.n_agents = get_n_agents()
        self.gamma = get_discount_factor()
        self.buffer = buffer
        self.action_dim = Converter.for_space(env.action_space).shape[0]

        self.value = JointCritic(env)
        self.value_target = JointCritic(env)

        self.optimizer = torch.optim.Adam(
            self.value.parameters(), lr=get_learning_rate_for_critic()
        )
        self.init_action_mapping()

    def init_action_mapping(self):
        self.n_joint_action = self.action_dim ** self.n_agents

        def actions2joint(a):
            # takes joint a and returns the combination in the joint space
            # actions are encoded into action_dim_decimal numbers with n_agent digits
            return sum(a * self.action_dim ** i for i, a in enumerate(a))

        def joint2actions(joint_action):
            # the inverse of actions2joint
            a = []
            for i in reversed(range(self.n_agents)):
                a.append(joint_action // (self.action_dim ** i))
                joint_action = joint_action % (self.action_dim ** i)
            assert joint_action == 0
            return tuple(reversed(a))

        # stores action tuples in the order of joint action indexes
        self.joint_action_table = []
        # stores joint action index in the shape of factored actions
        self.joint_index = torch.zeros([self.action_dim] * self.n_agents, dtype=int)
        for i in range(self.n_joint_action):
            a = joint2actions(i)
            # print(a)
            self.joint_action_table.append(a)
            self.joint_index[a] = i
            # print(self.joint_index)

        self.joint_action_table = np.array(self.joint_action_table)

    def forward(self, x: Tensor, target_val) -> Tensor:
        return self.value_target(x) if target_val else self.value(x)

    def actions2joints(self, A):
        return torch.stack([self.joint_index[tuple(a)] for a in A])

    def joint2actions(self, joint_action):
        return self.joint_action_table[joint_action]

    def policy2joint(self, policies):
        joint_policy = torch.zeros(self.n_joint_action)
        for j, a in enumerate(self.joint_action_table):
            joint_policy[j] = torch.prod(
                torch.tensor([policies[i][a] for i, a in enumerate(a)])
            )
        return joint_policy

    def add_agents(self, agents):
        self.agents = agents

    def get_value(self, traces, target_val):
        return self(traces.view(traces.shape[0], traces.shape[1], -1), target_val)

    def from_buffer(self, target_val=False, agent_id=None, advantage=False):
        # return values for exps in buffer
        # used in get_all_action_and_value which provides training targets
        traces, actions, _, _, _, states = self.buffer.get_trajectory()

        if get_use_predefined_value_func():
            assert get_is_q_func()
            return torch.tensor(
                [TrickyStagHunt.true_q(s, a) for s, a in zip(traces[:-1, -1], actions)]
            ).unsqueeze(1)

        input_data = self.get_input_data(states, traces)

        values = self.get_value(input_data[:-1], target_val)

        if not get_is_q_func():
            assert not advantage
            return values

        if not get_use_marginalized_q():
            return values.gather(1, self.actions2joints(actions).unsqueeze(-1))

        # use marginalized central Q case

        assert agent_id is not None and type(agent_id) is int
        policies = [
            a.action_probs(traces[:-1, :, i,]) for i, a in enumerate(self.agents)
        ]  # (agent, bs, actd) e.g. (2,800,2)
        policies = torch.stack(
            [self.policy2joint(p) for p in zip(*policies)]
        )  # (bs, joint_action_prob) e.g.(800,4)

        values_under_policy = (values * (policies * self.n_joint_action)).view(
            [-1] + [self.action_dim] * self.n_agents
        )

        state_values = values_under_policy.mean(dim=list(range(1, self.n_agents + 1)))
        marginalzied_values = values_under_policy.mean(
            dim=[i + 1 for i in range(self.n_agents) if i != agent_id]
        )

        res = marginalzied_values.gather(
            1, Tensor(actions)[:, agent_id].view(-1, 1).long()
        )
        if advantage:
            res -= state_values.unsqueeze(1)
        return res

    def get_input_data(self, states, traces):
        if get_critic_type() == 'state':
            input_data = states.unsqueeze(2)
        elif get_critic_type() == 'history':
            input_data = traces
        elif get_critic_type() == 'state+history':
            # expand state shape to match n
            state_shape = list(states.shape)
            state_shape[1] = get_trace_len()
            expanded_states = states.expand(*state_shape) # copies along trace_len dim
            # traces.shape: [n_step, trace_len, n_agent, obs_len]
            joint_traces = traces.reshape((traces.shape[0], traces.shape[1], -1))
            input_data = torch.cat((expanded_states, joint_traces), 2)
        else:
            raise ValueError('critic type {} invalid'.format(get_critic_type()))
        return input_data

    def learn(self):
        traces, actions, rewards, returns, dones, states = self.buffer.get_trajectory(
            joint_reward=True
        )

        input_data = self.get_input_data(states, traces)
        
        values = self.get_value(input_data[:-1], target_val=False)

        if not get_is_q_func():
            # assuming the value function update is full MC
            target = torch.tensor(returns).unsqueeze(1)
        else:
            values = values.gather(1, self.actions2joints(actions).unsqueeze(-1))
            next_values = (
                self.get_value(input_data[1:], target_val=True)
                .max(dim=1)
                .values.unsqueeze(1)
            )
            # set terminal states' values to 0
            next_values *= torch.tensor(1 - np.array(dones).reshape((-1, 1)), dtype=int)
            target = torch.tensor(rewards).unsqueeze(1) + self.gamma * next_values

        self.optimizer.zero_grad()
        loss = (values - target).pow(2).mean()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.value_target.load_state_dict(self.value.state_dict())
