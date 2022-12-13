import imageio
import os
import hydra
import gym
import d4rl
import numpy as np

from PIL import Image
from omegaconf import DictConfig, OmegaConf

from dreamer.agent import DreamerV2, HERDreamerV2, GCDreamer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    env = gym.make('maze2d-open-v0')

    EVAL_LOG_PATH = "/home/luyuan/thomaswe/16831_project/data/"
    RUN_NAME = "run0"
    ALGO = "GCDreamerV2_lower_kl"

    if "HER" in ALGO:
        agent = HERDreamerV2(cfg.agent, dict(vector=(4,), goal=(2,)), env.action_space.sample().shape[0], RUN_NAME)
        agent_path = "/home/luyuan/thomaswe/16831_project/outputs/2022-12-12/16-08-18"
        agent.load_networks(os.path.join(agent_path, "wm_25000.pth"), os.path.join(agent_path, "behavior_25000.pth"))
        print("HER AGENT")
    elif "GC" in ALGO:
        agent = GCDreamer(cfg.agent, dict(vector=(4,), goal=(2,)), env.action_space.sample().shape[0], RUN_NAME)
        agent_path = "/home/luyuan/thomaswe/16831_project/outputs/2022-12-12/19-07-57"
        agent.load_networks(os.path.join(agent_path, "wm_25000.pth"), os.path.join(agent_path, "behavior_25000.pth"))
        print("GC AGENT")
    else:
        agent = DreamerV2(cfg.agent, dict(vector=(4,), goal=(2,)), env.action_space.sample().shape[0], RUN_NAME)
        agent_path = "/home/luyuan/thomaswe/16831_project/outputs/2022-12-12/14-30-07"
        agent.load_networks(os.path.join(agent_path, "wm_25000.pth"), os.path.join(agent_path, "behavior_25000.pth"))
        print("VANILLA AGENT")

    frames = []
    for _ in range(5):
        agent.reset()
        action = np.zeros(env.action_space.shape)
        obs = env.reset()
        is_last = False
        while not is_last:
            frame = env.render(mode='rgb_array')
            frames.append(Image.fromarray(frame))
            action = agent.step(obs)
            obs, _, is_last, _ = env.step(action)
            if is_last:
                agent.reset()
                agent.log_episode()

    imageio.mimwrite(os.path.join(EVAL_LOG_PATH, f"{ALGO}_agent.gif"), frames, fps=60)

if __name__ == "__main__":
    main()