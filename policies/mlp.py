import torch
import torch.nn as nn
from utils.weights_init import init, init_normc_

class MLP(nn.Module):
    """docstring for MLP."""
    def __init__(self, obs_shape, hidden_size=64):
        super(MLP, self).__init__()

        self.n_inputs = obs_shape[0]
        self._hidden_size = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(self.n_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(self.n_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, 1))
        )


        # self.actor = nn.Sequential(
        #     nn.Linear(self.n_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU()
        # )
        #
        # self.critic = nn.Sequential(
        #     nn.Linear(self.n_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 1)
        # )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x):
        value = self.critic(x)
        action = self.actor(x)
        return value, action
