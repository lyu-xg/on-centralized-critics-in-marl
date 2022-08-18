from torch import nn

# these basic blocks are here to support a sequential shape of [batch_size, seq_length, obs_dim]
# with the exception of LinearWithTraces, which also supports [batch_size, obs_dim]


class LinearWithTraces(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.nonlinear = nn.Tanh()

    def forward(self, x):
        if len(x.shape) == 3:
            bs, tl, n = x.shape
            x = x.view(bs * tl, n)
            return self.nonlinear(self.linear(x)).view(bs, tl, self.output_dim)
        else:
            return self.nonlinear(self.linear(x))


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)
        self.nonlinear = nn.Tanh()

    def forward(self, x):
        # set batch size to 1 and discard the hidden states
        x, h = self.lstm(x)
        return self.nonlinear(x)


class Collect(nn.Module):
    """
        in the end of the layers, we are left with [bs, 1, n],
        so just take [bs, n].
    """

    def forward(self, x):
        size = x.size()
        if len(size) == 3:
            # assert size[1] == 1
            return x[:, -1]
        return x
