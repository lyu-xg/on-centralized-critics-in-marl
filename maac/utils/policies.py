import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=128, nonlin=F.leaky_relu,
                 norm_in=False, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X, H):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X.view(-1, X.shape[-1])).reshape(X.shape)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2, H = self.lstm(h1, H)
        #h2 = self.nonlin(self.fc2(h1))
        out = self.fc2(h2)
        return out, H


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, H=None, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False, valid=None):
        out, H = super(DiscretePolicy, self).forward(obs, H)
        out = out.reshape(-1, out.shape[-1])
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            # rets.append([(out**2).mean()])
            rets.append([((out**2).mean(-1, keepdim=True) * valid.view(-1,1)).sum() / valid.sum()])
        if return_entropy:
            # rets.append(-(log_probs * probs).sum(1).mean())
            rets.append(((-(log_probs * probs).sum(1, keepdim=True))*valid.view(-1,1)).sum() / valid.sum())
        if len(rets) == 1:
            return rets[0], H
        return rets, H
