import torch.nn as nn, torch.nn.functional as F, torch.distributions as D, torch.nn.init as init

from basalganglia.reinforce.util.torch_util import *


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

        self.fc_layer = nn.Linear(hidden_layer_width, hidden_layer_width)

        if type(self.action_space) is Discrete:
            self.logit = nn.Linear(hidden_layer_width, self.action_space.n)
        elif type(self.action_space) is Box:
            self.action_min, self.action_max = env.env.action_space.low, env.env.action_space.high
            self.mean = nn.Linear(hidden_layer_width, self.action_space.shape[0])
            self.log_sigma = nn.Parameter(torch.ones(self.action_space.shape[0])*self.init_log_sigma)
            init.zeros_(self.mean.weight)
        else:
            raise NotImplementedError

    def forward(self, s, detach=False):
        x = self.input_layer(s)
        x = torch.tanh(x)
        x = self.fc_layer(x)
        x = torch.tanh(x)

        if type(self.action_space) is Discrete:
            p = self.logit(x)
            if detach:
                p = p.detach()
            d = D.Categorical(logits=p)
            return d
        elif type(self.action_space) is Box:
            mu = self.mean(x)
            sigma = torch.exp(self.log_sigma)   # do softplus?
            if detach:
                mu, sigma = mu.detach(), sigma.detach()
            d = D.MultivariateNormal(mu, torch.diag(sigma))
            return d

    def policy(self, s):
        state = torchify([s], type(self.state_space))
        action = self.forward(state).sample()
        return action.numpy()[0]
