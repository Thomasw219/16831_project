import imageio
from PIL import Image
import gym
import d4rl # Import required to register environments

import matplotlib.pyplot as plt

# Create the environment
env = gym.make('antmaze-umaze-diverse-custom-v2')

# d4rl abides by the OpenAI gym interface

total_rewards = []
for _ in range(100):
    obs = env.reset()
    total_reward = 0
    for i in range(1000):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward

    total_rewards.append(total_reward)

plt.scatter(range(len(total_rewards)), total_rewards)
plt.ylabel("Episode Returns")
plt.xlabel("Random goal (individual episodes)")
plt.savefig("./data/fig.png")