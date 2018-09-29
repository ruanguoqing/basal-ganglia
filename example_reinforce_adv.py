import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim

from trainer.training_loop import *


hype = {'Environment Name': 'CartPole-v0',
        'Learning Algorithm': 'Reinforce with Advantages',
        'Policy Network': {'Hidden Layer Width': 128},
        'Value Network': {'Hidden Layer Width': 128},
        'Buffer': {'Buffer Length': 1},
        'Policy Optimizer': {'Algorithm': optim.Adam, 'Learning Rate': 1e-3},
        'Value Optimizer': {'Algorithm': optim.Adam, 'Learning Rate': 1e-3},
        'Episodes per Step': 1,
        'Number of Steps': 10000,
        'Regularization Parameter': 0.1,
        'Discount': 0.99,
        'Log Every': 100,
        'Break at Reward': 195}

perf_profile = []
for (s, e, r) in learn(hype):
    print('Steps: %04d,\tEpisodes: %04d,\tReward per Episode:%f' % (s, e, r))
    perf_profile.append((s, e, r))

s, e, r = map(list, zip(*perf_profile))
df = pd.DataFrame(dict(episode=e, reward=r))
sns.relplot(x='episode', y='reward', data=df, kind='line')
plt.show()