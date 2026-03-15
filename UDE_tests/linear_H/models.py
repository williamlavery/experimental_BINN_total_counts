import torch.nn as nn
import torch


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class SolutionNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_weights_xavier)

    def forward(self, t):
        return self.net(t)


class DynamicsNet(nn.Module):
    """Linear model for H(N) = a * N + b.

    Initialized with a negative slope so H starts with a decreasing trend in N.
    """

    def __init__(self, hidden_dim=4, init_slope=-0.8, init_bias=0.8):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(init_slope)
            self.linear.bias.fill_(init_bias)

    def forward(self, N):
        return self.linear(N)


class SigmaNet(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, N):
        return self.net(N) + 1e-4
