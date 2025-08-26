import numpy as np
import matplotlib.pyplot as plt
from utils import moving_average
import pickle

filename = 'returns.pkl'
with open(filename, 'rb') as f:
    rewards = pickle.load(f)


episodes_list = list(range(len(rewards)))

# plot the training rewards
mv_rewards = moving_average(rewards, 100)
plt.plot(episodes_list, mv_rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('rl_train_rewards.png', dpi=150, bbox_inches='tight')
plt.show()

