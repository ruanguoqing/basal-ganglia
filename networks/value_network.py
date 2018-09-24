import torch.nn as nn, torch.nn.functional as F

from util.torch_util import *


class ValueNetwork(nn.Module):
    def __init__(self, env, hidden_layer_width=128):
        super(ValueNetwork, self).__init__()
        self.state_space, self.action_space = env.env.observation_space, env.env.action_space

        if type(self.state_space) is Discrete:
            self.input_layer = nn.Embedding(self.state_space.n, hidden_layer_width)
        elif type(self.state_space) is Box:
            self.input_layer = nn.Linear(self.state_space.shape[0], hidden_layer_width)
        else:
            raise NotImplementedError

        self.output_layer = nn.Linear(hidden_layer_width, 1)

    def forward(self, s, a=None):
        x = self.input_layer(s)
        x = F.relu(x)
        x = self.output_layer(x)
        x = x.squeeze(dim=-1)
        return x
