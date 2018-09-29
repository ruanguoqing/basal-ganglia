import torch.nn.functional as F, torch.distributions as D, torch.autograd as A

from util.torch_util import *


def npg_adv_step(policy_network, value_network, trace_summary, reg, opt_policy, opt_value):
    s_list = torchify(trace_summary[0], type(policy_network.state_space))
    a_list = torchify(trace_summary[1], type(policy_network.action_space))
    cum_r_list = torchify(trace_summary[3])

    val_list = value_network(s_list)
    cost_value = F.mse_loss(val_list, cum_r_list)

    d_list_old = policy_network(s_list, detach=True)
    logp_list_old = d_list_old.log_prob(a_list)
    d_list = policy_network(s_list)
    logp_list, ent_list = d_list.log_prob(a_list), d_list.entropy()

    value_policy = torch.mean(torch.exp(logp_list-logp_list_old) * (cum_r_list-val_list))
    entropy = torch.mean(ent_list)
    cost_policy = - value_policy - reg * entropy

    grad_cost = A.grad(cost_policy, policy_network.parameters(), retain_graph=True)
    grad_cost_flat = make_flat_from(grad_cost).data
    kl = torch.mean(D.kl_divergence(d_list_old, d_list))
    fvp = lambda v: hvp(kl, policy_network.parameters, v)
    natural_grad_flat = do_conjugate_gradient(fvp, grad_cost_flat, n_iters=10)
    normalized_step = torch.sqrt((grad_cost_flat * natural_grad_flat).sum())
    #print(normalized_step, torch.norm(grad_cost_flat), torch.norm(natural_grad_flat))
    opt_policy.zero_grad()
    cost_policy.backward(retain_graph=True)
    set_grad_from_flat(policy_network, natural_grad_flat/normalized_step)
    opt_policy.step()

    opt_value.zero_grad()
    cost_value.backward()
    opt_value.step()

