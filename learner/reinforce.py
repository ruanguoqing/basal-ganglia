import torch.nn.functional as F

from util.torch_util import *


def reinforce_mc_step(policy_network, trace_summary, reg, opt):
    s_list = torchify(trace_summary[0], type(policy_network.state_space))
    a_list = torchify(trace_summary[1], type(policy_network.action_space))
    cum_r_list = torchify(trace_summary[3])

    d_list = policy_network(s_list)
    logp_list, ent_list = d_list.log_prob(a_list), d_list.entropy()

    value_policy = torch.mean(cum_r_list * logp_list)
    entropy = torch.mean(ent_list)
    cost = - value_policy - reg * entropy

    opt.zero_grad()
    cost.backward()
    opt.step()


def reinforce_adv_step(policy_network, value_network, trace_summary, reg, opt_policy, opt_value):
    s_list = torchify(trace_summary[0], type(policy_network.state_space))
    a_list = torchify(trace_summary[1], type(policy_network.action_space))
    cum_r_list = torchify(trace_summary[3])

    val_list = value_network(s_list)
    cost_value = F.mse_loss(val_list, cum_r_list)

    d_list = policy_network(s_list)
    logp_list, ent_list = d_list.log_prob(a_list), d_list.entropy()

    value_policy = torch.mean((cum_r_list-val_list) * logp_list)
    entropy = torch.mean(ent_list)
    cost_policy = - value_policy - reg * entropy

    opt_policy.zero_grad()
    cost_policy.backward(retain_graph=True)
    opt_policy.step()

    opt_value.zero_grad()
    cost_value.backward()
    opt_value.step()

