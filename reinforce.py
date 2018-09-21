import torch, torch.nn.functional as F

from torch_util import *


def reinforce_mc_step(policy_network, trace_summary, reg, opt):
    s_list, a_list, _, cum_r_list = map(torchify, trace_summary)

    logit_list = policy_network.logit(s_list)
    p_list, logp_list = F.softmax(logit_list, dim=-1), F.log_softmax(logit_list, dim=-1)
    logp_actions = torch.sum(logp_list * make_indicator(a_list), dim=1)

    value_policy = torch.mean(cum_r_list * logp_actions)
    entropy = -torch.mean(logp_list * p_list)
    cost = -value_policy - reg * entropy

    opt.zero_grad()
    cost.backward()
    opt.step()