import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
her_td3_means = np.load("./data/her_td3_reward_data.npz")['means']
ppo_means = np.load("./data/ppo_reward_data.npz")['means']
td3_only_means = np.load("./data/td3_only_reward_data.npz")['means']
random_mean = 4.56

t = np.arange(her_td3_means.shape[0]) * 10000
plt.plot(t, np.ones_like(t) * random_mean, label="Random Agent")
plt.plot(t, her_td3_means, label="TD3 with HER")
plt.plot(t, td3_only_means, label="TD3")
plt.plot(t, ppo_means, label="PPO")
plt.legend()
plt.ylabel("Average Rewards")
plt.xlabel("Environment Samples")
plt.savefig("./data/performance.png")
