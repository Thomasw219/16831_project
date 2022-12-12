import gym
import d4rl
import numpy as np


env = gym.make('maze2d-open-v0')
all_goals = []
for i in range(1000):
    desired_goal = env.reset()["desired_goal"]
    all_goals.append(desired_goal)

all_goals = np.array(all_goals)
print(np.max(all_goals[:, 0]), np.min(all_goals[:, 0]))
print(np.max(all_goals[:, 1]), np.min(all_goals[:, 1]))
print(np.mean(all_goals))
print(np.std(all_goals))