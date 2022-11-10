import imageio
from PIL import Image
import gym
import d4rl # Import required to register environments
import numpy as np

import matplotlib.pyplot as plt

# Create the environment
env = gym.make('maze2d-open-v0')

# d4rl abides by the OpenAI gym interface

total_rewards = []
for _ in range(100):
    obs = env.reset()
    total_reward = 0
    for i in range(150):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward

    total_rewards.append(total_reward)

print(np.mean(total_rewards), np.std(total_rewards))