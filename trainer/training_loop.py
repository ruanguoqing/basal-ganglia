import torch.optim as optim

from util.game import *
from trainer.replay_buffer import *
from networks.policy_network import *
from networks.value_network import *
from learner.reinforce import *


def learn(hype):
    env = Game(hype['Environment Name'])
    replay_buffer = ReplayBuffer(hype['Buffer']['Buffer Length'])
    policy_network = PolicyNetwork(env, hype['Policy Network']['Hidden Layer Width'])
    opt_policy = optim.Adam(policy_network.parameters(), lr=hype['Policy Optimizer']['Learning Rate'])
    if hype['Learning Algorithm'] == 'Reinforce with Advantages':
        value_network = ValueNetwork(env, hype['Value Network']['Hidden Layer Width'])
        opt_value = optim.Adam(value_network.parameters(), lr=hype['Value Optimizer']['Learning Rate'])

    reward_past_few = []
    for i_iter in range(hype['Number of Steps']):
        trace_eps = env.play_episodes(policy_network.policy, n_eps=hype['Episodes per Step'])
        replay_buffer.absorb_trace(trace_eps)
        trace_summary = replay_buffer.emit_trace_as_grouped(hype['Discount'])

        reward_past_few.append(sum(trace_summary[2]) / hype['Episodes per Step'])
        if not i_iter % hype['Log Every']:
            reward_run_avg = sum(reward_past_few) / len(reward_past_few)
            yield (i_iter, i_iter * hype['Episodes per Step'], reward_run_avg)
            if reward_run_avg > hype['Break at Reward']: break
            reward_past_few = []

        if hype['Learning Algorithm'] == 'Reinforce with MC':
            reinforce_mc_step(policy_network, trace_summary, hype['Regularization Parameter'], opt_policy)
        elif hype['Learning Algorithm'] == 'Reinforce with Advantages':
            reinforce_adv_step(policy_network, value_network, trace_summary,
                               hype['Regularization Parameter'], opt_policy, opt_value)
        else:
            raise NotImplementedError
