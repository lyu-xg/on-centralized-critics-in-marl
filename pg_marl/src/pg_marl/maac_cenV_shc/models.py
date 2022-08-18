import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc

class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_layer_size=64, rnn_layer_size=64):
        super(Actor, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size, act_fn='leaky_relu')
        self.lstm = nn.LSTM(rnn_layer_size, hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc2 = Linear(mlp_layer_size, output_dim, act_fn='linear')

    def forward(self, x, h=None, eps=0.0, test_mode=False):

        x = F.leaky_relu(self.fc1(x))
        x, h = self.lstm(x, h)
        x = self.fc2(x)

        action_logits = F.log_softmax(x, dim=-1)

        if not test_mode:
            logits_1 = action_logits + np.log(1-eps)
            logits_2 = torch.full_like(action_logits, np.log(eps)-np.log(action_logits.size(-1)))
            logits = torch.stack([logits_1, logits_2])
            action_logits = torch.logsumexp(logits,axis=0)

        return action_logits, h

class Critic(nn.Module):

    def __init__(self, state_input_dim, obs_input_dim, output_dim=1, mlp_layer_size=64, rnn_layer_size=64):
        super(Critic, self).__init__()

        self.fc1 = Linear(state_input_dim, mlp_layer_size, act_fn='leaky_relu')
        self.fc2 = Linear(obs_input_dim, mlp_layer_size, act_fn='leaky_relu')
        self.lstm = nn.LSTM(mlp_layer_size, hidden_size=rnn_layer_size, num_layers=1, batch_first=True)
        self.fc3 = Linear(mlp_layer_size+rnn_layer_size, output_dim, act_fn='linear')

    def forward(self, s, obs, h=None):

        s_x = F.leaky_relu(self.fc1(s))
        obs_x = F.leaky_relu(self.fc2(obs))
        obs_x, h = self.lstm(obs_x, h)
        state_value = self.fc3(torch.cat([s_x, obs_x], dim=-1))
        return state_value, h 
