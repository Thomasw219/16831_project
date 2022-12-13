import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
her_td3_means = np.load("./data/her_td3_reward_data.npz")['means']
ppo_means = np.load("./data/ppo_reward_data.npz")['means']
td3_only_means = np.load("./data/td3_only_reward_data.npz")['means']
dreamerv2_means = np.load("./data/DreamerV2_lower_kl_reward_data.npz")['means']
her_dreamerv2_means = np.load("./data/HERDreamerV2_lower_kl_reward_data.npz")['means']
gc_dreamerv2_means = np.load("./data/GCDreamerV2_lower_kl_reward_data.npz")['means']
random_mean = 4.56

t = np.arange(her_td3_means.shape[0]) * 10000
plt.plot(t, np.ones_like(t) * random_mean, label="Random Agent")
plt.plot(np.arange(her_td3_means.shape[0]) * 10000, her_td3_means, label="TD3 with HER")
plt.plot(np.arange(td3_only_means.shape[0]) * 10000, td3_only_means, label="TD3")
plt.plot(np.arange(ppo_means.shape[0]) * 10000, ppo_means, label="PPO")
plt.plot(np.arange(dreamerv2_means.shape[0]) * 10000, dreamerv2_means, label="DreamerV2")
plt.plot(np.arange(her_dreamerv2_means.shape[0]) * 10000, her_dreamerv2_means, label="Modification 1")
plt.plot(np.arange(gc_dreamerv2_means.shape[0]) * 10000, gc_dreamerv2_means, label="Modification 2")
plt.legend()
plt.ylabel("Average Rewards")
plt.xlabel("Environment Samples")
plt.savefig("./data/performance.png")
