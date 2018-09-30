import numpy as np
import gym

from basalganglia.trajopt.environments.util import *
from basalganglia.trajopt.environments.continuous_mountain_car import *


env = Continuous_MountainCarEnv()
state = env.reset()

# How to run a single episode.
is_done = False
while not is_done:
    action = [np.random.normal()]
    state, reward, is_done, _ = env.step(action)

# Getting reward function and such
r, s, a = env.get_reward_function([-0.2, 0.1], [0.1])
print('Reward is ', r.detach().numpy())
print('d(Reward)/ds is ', grad(r, s))

f, s, a = env.get_transition_function([-0.2, 0.1], [0.1])
print('f(s,a) is ', f.detach().numpy())
