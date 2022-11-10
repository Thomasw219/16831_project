import imageio
import gym
import d4rl

from PIL import Image
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import HerReplayBuffer, TD3, PPO

env = gym.make('maze2d-open-v0')
env = Monitor(env)

# ALGO = "her_td3"
# model_path = "./models/td3_step49"
# model = TD3.load(model_path, env=env)

# ALGO = "ppo"
# model_path = "./models/ppo_step49"
# model = PPO.load(model_path, env=env)

ALGO = "td3_only"
model_path = "./models/td3_only_step49"
model = TD3.load(model_path, env=env)


frames = []
for _ in range(5):
    obs = env.reset()
    for _ in range(150):
        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

env.close()
imageio.mimwrite(f"./data/{ALGO}_agent.gif", frames, fps=60)
