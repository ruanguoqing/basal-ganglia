import numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch_util import *


class PolicyNetwork:
    def __init__(self, env, hidden_layer_width=128):
        self.input_dim, self.output_dim = env.state_info[1], env.action_info[1]
        self.logit = nn.Sequential(nn.Linear(self.input_dim, hidden_layer_width),
                                   nn.ReLU(),
                                   nn.Linear(hidden_layer_width, self.output_dim))

    def policy(self, s):
        state = torchify([s])
        prob = F.softmax(self.logit(state), dim=-1).detach().numpy()[0]
        return np.random.choice(self.output_dim, p=prob)
