import imageio
from PIL import Image
import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('maze2d-open-v0')

frames = []
for _ in range(5):
    env.reset()
    for _ in range(150):
        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))

        env.step(env.action_space.sample())

env.close()
imageio.mimwrite("./data/random_agent.gif", frames, fps=60)
