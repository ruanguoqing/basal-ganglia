import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim

from basalganglia.reinforce.trainer.training_loop import *


hype = {'Environment Name': 'Swimmer-v2',
        'Learning Algorithm': 'Natural Policy Gradient with Advantages',
        'Policy Network': {'Hidden Layer Width': 32},
        'Value Network': {'Hidden Layer Width': 32},
        'Buffer': {'Buffer Length': 1},
        'Policy Optimizer': {'Algorithm': optim.SGD, 'Learning Rate': 1e-1},
        'Value Optimizer': {'Algorithm': optim.Adam, 'Learning Rate': 1e-3},
        'Episodes per Step': 10,
        'Number of Steps': 50,
        'Regularization Parameter': 0.1,
        'Discount': 0.995,
        'Log Every': 1,
        'Break at Reward': 195}

perf_profile = []
print("%s with %s" % (hype['Environment Name'], hype['Learning Algorithm']))
for (s, e, r) in learn(hype):
    print('Steps: %04d,\tEpisodes: %04d,\tReward per Episode:%f' % (s, e, r))
    perf_profile.append((s, e, r))

s, e, r = map(list, zip(*perf_profile))
df = pd.DataFrame(dict(episode=e, reward=r))
sns.relplot(x='episode', y='reward', data=df, kind='line')
plt.show()
