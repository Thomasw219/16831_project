import numpy as np
import torch
from collections import deque

class Memory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_episodes = deque(maxlen=int(self.cfg.max_episodes))
        self.train_timesteps = 0
        self.test_ratio = cfg.test_ratio
        self.test_episodes = deque(maxlen=int(self.cfg.max_episodes))
        self.test_timesteps = 0
        self.last_episode = None
        self.reset_episode()

    def reset_episode(self):
        self.current_episode = dict(
            observation=[],
            action=[],
            reward=[],
            is_first=[],
            is_last=[],
        )

    def last_episode_metrics(self):
        if self.last_episode is None:
            return {}
        else:
            episode = self.last_episode
            return dict(
                length=episode['actions'].shape[0],
                returns=np.sum(episode['rewards'])
            )

    def step(self, **kwargs):
        for k, v in kwargs.items():
            self.current_episode[k].append(v)

        if kwargs['is_last'][0]:
            length = len(self.current_episode['action'])
            if length >= self.cfg.min_episode_len:
                episode = dict(
                    observations = {k: np.stack([np.array(t[k], dtype=np.float16) for t in self.current_episode['observation']]) for k in self.current_episode['observation'][0].keys()},
                    actions = np.array(self.current_episode['action'], dtype=np.float16),
                    rewards = np.array(self.current_episode['reward'], dtype=np.float16),
                    is_firsts = np.array(self.current_episode['is_first'], dtype=np.float16),
                    is_lasts = np.array(self.current_episode['is_last'], dtype=np.float16),
                )
                self.last_episode = episode
                if self.test_ratio != 0 and (self.test_timesteps == 0 or (self.test_timesteps) / (self.test_timesteps + self.train_timesteps) < self.test_ratio):
                    self.test_episodes.append(episode)
                    self.test_timesteps += length
                else:
                    self.train_episodes.append(episode)
                    self.train_timesteps += length
            self.reset_episode()

    def sample(self, episodes, device):
        batch_episodes = np.random.choice(episodes, self.cfg.B, replace=True)
        batch_goals = []
        batch_vectors = []
        batch_actions = []
        batch_rewards = []
        batch_is_firsts = []
        batch_is_lasts = []

        for episode in batch_episodes:
            length = episode['actions'].shape[0]
            t_s = np.clip(np.random.randint(0, length), 0, length - self.cfg.T - 1)
            t_f = t_s + self.cfg.T
            batch_goals.append(episode['observations']['desired_goal'][t_s: t_f])
            batch_vectors.append(episode['observations']['observation'][t_s: t_f])
            batch_actions.append(episode['actions'][t_s: t_f])
            batch_rewards.append(episode['rewards'][t_s: t_f])
            batch_is_firsts.append(episode['is_firsts'][t_s: t_f])
            batch_is_lasts.append(episode['is_lasts'][t_s: t_f])

        return dict(
            observations=dict(
                goals=torch.tensor(np.stack(batch_goals, axis=1), device=device),
                vectors=torch.tensor(np.stack(batch_vectors, axis=1), device=device),
            ),
            actions=torch.tensor(np.stack(batch_actions, axis=1), device=device),
            rewards=torch.tensor(np.stack(batch_rewards, axis=1), device=device),
            is_firsts=torch.tensor(np.stack(batch_is_firsts, axis=1), device=device),
            is_lasts=torch.tensor(np.stack(batch_is_lasts, axis=1), device=device),
        )

    def train_sample(self, device=torch.device("cpu")):
        return self.sample(self.train_episodes, device)

    def test_sample(self, device=torch.device("cpu")):
        return self.sample(self.test_episodes, device)

    def get_metrics(self):
        return dict(
            train_episodes=len(self.train_episodes),
            train_timesteps=self.train_timesteps,
            test_episodes=len(self.test_episodes),
            test_timesteps=self.test_timesteps,
        )

class HERMemory(Memory):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.future_ratio = 0.5
        self.pos_mean = np.array([1.5, 2.5])
        self.pos_std = np.array([3, 5])
        self.goal_max = np.array([0.53, 0.51])
        self.goal_min = np.array([-0.19, -0.31])

    def compute_reward(self, achieved_goal, desired_goal):
        achieved_goal = achieved_goal * self.pos_std + self.pos_mean
        desired_goal = desired_goal * self.pos_std + self.pos_mean
        reward = 1.0 if np.linalg.norm(desired_goal - achieved_goal) <= 0.5 else 0.0
        return reward

    def sample(self, episodes, device):
        batch_episodes = np.random.choice(episodes, self.cfg.B, replace=True)
        her_future_goal = np.random.uniform(0.0, 1.0, size=(len(batch_episodes),)) < self.future_ratio
        batch_goals = []
        batch_vectors = []
        batch_actions = []
        batch_rewards = []
        batch_is_firsts = []
        batch_is_lasts = []

        for (episode, use_future_goal) in zip(batch_episodes, her_future_goal):
            length = episode['actions'].shape[0]
            t_s = np.clip(np.random.randint(0, length), 0, length - self.cfg.T - 1)
            t_f = t_s + self.cfg.T
            if use_future_goal:
                future_index = np.random.randint(t_s, length - 1)
                future_goal = episode['observations']['achieved_goal'][future_index]
                batch_goals.append(np.stack([future_goal for _ in range(self.cfg.T)], axis=0))
                batch_rewards.append(np.array([[self.compute_reward(episode['observations']['achieved_goal'][t_s + i], future_goal)] for i in range(self.cfg.T)]))
            else:
                sampled_goal = np.random.uniform(self.goal_min, self.goal_max)
                batch_goals.append(np.stack([sampled_goal for _ in range(self.cfg.T)], axis=0))
                batch_rewards.append(np.array([[self.compute_reward(episode['observations']['achieved_goal'][t_s + i], sampled_goal)] for i in range(self.cfg.T)]))
            batch_vectors.append(episode['observations']['observation'][t_s: t_f])
            batch_actions.append(episode['actions'][t_s: t_f])
            batch_is_firsts.append(episode['is_firsts'][t_s: t_f])
            batch_is_lasts.append(episode['is_lasts'][t_s: t_f])

        return dict(
            observations=dict(
                goals=torch.tensor(np.stack(batch_goals, axis=1), device=device, dtype=torch.float32),
                vectors=torch.tensor(np.stack(batch_vectors, axis=1), device=device),
            ),
            actions=torch.tensor(np.stack(batch_actions, axis=1), device=device),
            rewards=torch.tensor(np.stack(batch_rewards, axis=1), device=device, dtype=torch.float32),
            is_firsts=torch.tensor(np.stack(batch_is_firsts, axis=1), device=device),
            is_lasts=torch.tensor(np.stack(batch_is_lasts, axis=1), device=device),
        )