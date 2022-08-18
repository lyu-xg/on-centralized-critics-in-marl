import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shapes = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shapes[0], 128)
        self.fc2 = nn.Linear(input_shapes[1], 128)
        self.rnn = nn.GRU(128, 128, batch_first=True)
        self.fc3 = nn.Linear(128*2, self.n_actions)

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        # permute x_1 to match rnn sequence for each agent
        bs, tr, nag, dim = inputs[0].shape
        x_1 = F.leaky_relu(self.fc1(inputs[0].permute(0,2,1,3).reshape(-1, tr, dim)))
        x_1, h = self.rnn(x_1)
        x_1 = x_1.reshape(bs,nag,tr,-1).permute(0,2,1,3)
        x_2 = F.leaky_relu(self.fc2(inputs[1]))
        x = th.cat([x_1, x_2], dim=-1)
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observation
        inputs.append(th.cat([batch["obs"][:, ts].reshape(bs, max_t, 1, -1) for _ in range(self.n_agents)],
                              dim=-2))

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        other_agent_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)

        if self.args.state_critic:
            return (inputs, th.cat([other_agent_actions, batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)], dim=-1))
        else:
            return (inputs, other_agent_actions)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"] * self.n_agents
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # agent id
        input_shape += self.n_agents

        if self.args.state_critic:
            input_shapes = [input_shape, scheme["actions_onehot"]["vshape"][0] * self.n_agents + scheme["state"]["vshape"]]
        else:
            input_shapes = [input_shape, scheme["actions_onehot"]["vshape"][0] * self.n_agents]

        return input_shapes
