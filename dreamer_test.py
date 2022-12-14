import os
import hydra
import gym
import d4rl
import tqdm
import numpy as np

from omegaconf import DictConfig, OmegaConf

from dreamer.agent import DreamerV2, HERDreamerV2, GCDreamer

def train_agent(agent: DreamerV2, env, n_steps):
    steps_elapsed = 0
    while steps_elapsed < n_steps:
        agent.reset()
        action = np.zeros(env.action_space.shape)
        obs = env.reset()
        reward = 0.0
        is_first = True
        is_last = False
        agent.record(action=action, observation=obs, reward=[reward], is_first=[is_first], is_last=[is_last])
        episode_steps = 0
        while not is_last:
            episode_steps += 1
            steps_elapsed += 1
            is_first = False
            action = agent.step(obs)
            obs, reward, is_last, _ = env.step(action)
            agent.record(action=action, observation=obs, reward=[reward], is_first=[is_first], is_last=[is_last])
            if is_last:
                agent.reset()
                agent.log_episode()
        agent.train_joint(episode_steps // 4)

def eval_agent(agent, env, n_episodes):
    all_returns = []
    for _ in range(n_episodes):
        agent.reset()
        action = np.zeros(env.action_space.shape)
        obs = env.reset()
        returns = 0.0
        is_last = False
        while not is_last:
            action = agent.step(obs)
            obs, reward, is_last, _ = env.step(action)
            returns += reward
            if is_last:
                agent.reset()
                agent.log_episode()
        all_returns.append(returns)
    return np.mean(all_returns), np.std(all_returns)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    env = gym.make('maze2d-open-v0')

    TOTAL_STEPS = 500000
    EVAL_EVERY = 10000
    N_EVAL_EPISODES = 100
    EVAL_LOG_PATH = "/home/luyuan/thomaswe/16831_project/data/"
    LOG_DIR = "./logs/"
    RUN_NAME = "run0"
    ALGO = "GCDreamerV2_lower_kl"

    if "HER" in ALGO:
        agent = HERDreamerV2(cfg.agent, dict(vector=(4,), goal=(2,)), env.action_space.sample().shape[0], RUN_NAME)
        print("HER AGENT")
    elif "GC" in ALGO:
        agent = GCDreamer(cfg.agent, dict(vector=(4,), goal=(2,)), env.action_space.sample().shape[0], RUN_NAME)
        print("GC AGENT")
    else:
        agent = DreamerV2(cfg.agent, dict(vector=(4,), goal=(2,)), env.action_space.sample().shape[0], RUN_NAME)
        print("VANILLA AGENT")

    init_mean, init_std = eval_agent(agent, env, N_EVAL_EPISODES)
    # init_mean, init_std = 0, 0

    print_rewards = lambda means, stds: print(f"Reward means: {means[-1]} Stds {stds[-1]}")

    reward_means = [init_mean]
    reward_stds = [init_std]
    print_rewards(reward_means, reward_stds)
    np.savez(os.path.join(EVAL_LOG_PATH, f"{ALGO}_reward_data"), means=reward_means, stds=reward_stds)

    for step in range(np.ceil(TOTAL_STEPS / EVAL_EVERY).astype(int)):
        print(f"Step: {step}")
        train_agent(agent, env, EVAL_EVERY)
        agent.save_networks(f"{RUN_NAME}_{step}")
        mean, std = eval_agent(agent, env, N_EVAL_EPISODES)
        reward_means.append(mean)
        reward_stds.append(std)
        print_rewards(reward_means, reward_stds)
        np.savez(os.path.join(EVAL_LOG_PATH, f"{ALGO}_reward_data"), means=reward_means, stds=reward_stds)

if __name__ == "__main__":
    main()