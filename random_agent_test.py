import imageio
from PIL import Image
import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('antmaze-umaze-diverse-v2')

# d4rl abides by the OpenAI gym interface
env.reset()

frames = []
for i in range(1000):
    frame = env.render(mode='rgb_array')
    frames.append(Image.fromarray(frame))

    env.step(env.action_space.sample())

env.close()
imageio.mimwrite("./data/random_agent.gif", frames, fps=60)