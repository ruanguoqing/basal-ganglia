import torch.nn as nn, torch.nn.functional as F, torch.distributions as D, torch.nn.init as init

from util.torch_util import *


class PolicyNetwork(nn.Module):
    def __init__(self, env, hidden_layer_width=128, init_log_sigma=0, min_log_sigma=-3):
        super(PolicyNetwork, self).__init__()
        self.state_space, self.action_space = env.env.observation_space, env.env.action_space
        self.init_log_sigma, self.min_log_sigma = init_log_sigma, min_log_sigma

        if type(self.state_space) is Discrete:
            self.input_layer = nn.Embedding(self.state_space.n, hidden_layer_width)
        elif type(self.state_space) is Box:
            self.input_layer = nn.Linear(self.state_space.shape[0], hidden_layer_width)
        else:
            raise NotImplementedError

        if type(self.action_space) is Discrete:
            self.logit = nn.Linear(hidden_layer_width, self.action_space.n)
        elif type(self.action_space) is Box:
            self.action_min, self.action_max = env.env.action_space.low, env.env.action_space.high
            self.mean = nn.Linear(hidden_layer_width, self.action_space.shape[0])
            self.log_sigma = nn.Parameter(torch.ones(self.action_space.shape[0])*self.init_log_sigma)
            init.zeros_(self.mean.weight)
        else:
            raise NotImplementedError

    def forward(self, s, a=None):
        x = self.input_layer(s)
        x = F.relu(x)

        if type(self.action_space) is Discrete:
            p = self.logit(x)
            d = D.Categorical(logits=p)
            a = d.sample() if a is None else a
            return a, d.log_prob(a), d.entropy()
        elif type(self.action_space) is Box:
            mu = self.mean(x)
            std = torch.exp(self.log_sigma)#+self.min_log_sigma   # do softplus?
            d = D.MultivariateNormal(mu, torch.diag(std))
            a = d.sample() if a is None else a
            return a, d.log_prob(a), d.entropy()

    def policy(self, s):
        state = torchify([s], type(self.state_space))
        action, _, _ = self.forward(state, a=None)
        return action.numpy()[0]
