import gym
import d4rl # Import required to register environments
import os
import numpy as np

from stable_baselines3 import HerReplayBuffer, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

TOTAL_STEPS = 500000
EVAL_EVERY = 10000
N_EVAL_EPISODES = 100
EVAL_LOG_PATH = "./data/"
LOG_DIR = "./logs/"
RUN_NAME = "run0"

# Create the environment
env = gym.make('maze2d-open-v0')
env = Monitor(env)

model = TD3(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        online_sampling=True,
        max_episode_length=700,
    ),
    verbose=1,
    tensorboard_log=LOG_DIR,
)

eval = lambda model: evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)
init_mean, init_std = eval(model)

print_rewards = lambda means, stds: print(f"Reward means: {means[-1]} Stds {stds[-1]}")

reward_means = [init_mean]
reward_stds = [init_std]
print_rewards(reward_means, reward_stds)

for step in range(np.ceil(TOTAL_STEPS / EVAL_EVERY).astype(int)):
    print(f"Step: {step}")
    model.learn(
        EVAL_EVERY,
        progress_bar=True,
        tb_log_name=RUN_NAME,
        )
    model.save(f"./models/td3_step{step}")
    mean, std = eval(model)
    reward_means.append(mean)
    reward_stds.append(std)
    print_rewards(reward_means, reward_stds)
    np.savez(os.path.join(EVAL_LOG_PATH, "reward_data"), means=reward_means, stds=reward_stds)
