import sys
import os
import warnings
from pathlib import Path
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np

ROOT_PATH = Path(__file__).parents[1].absolute()
sys.path.append(str(ROOT_PATH))

from network.net import Actor
from network.encoder import get_encoder
from smarts_env.env import make_env

from config.config import load_config
config = load_config('config.yaml', 'config')

state_dim = load_config('encoder.yaml', 'encoder').dim

# warnings.filterwarnings("ignore", message="vehicle id named .* is more than 50 characters long.*")
warnings.filterwarnings("ignore")


def eval(rounds: int, encoder_path: str, policy_path: str):
    env = make_env(config.env_mode, config.max_episode_steps, config.seed, config.eval_headless, config.reward_mode)

    encoder = get_encoder(image_size=env.observation_space.shape[1], channels=env.observation_space.shape[0])
    encoder.load_state_dict(torch.load(encoder_path))
    encoder = encoder.eval()

    policy = Actor(state_dim, env.action_space.shape[0])
    policy.load_state_dict(torch.load(policy_path))
    policy = policy.eval()

    print("="*20, "Evaluating", "="*20)
    eval_sum_reward = 0
    for j in range(rounds):
        sum_reward = 0
        obs, _ = env.reset()
        obs = obs / 255.0
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        state = encoder(obs)
        print("*"*20, f"Round {j + 1} ", "*"*20)
        while True:
            mu, log_sigma =policy(state)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            action = [dist.sample().squeeze().numpy(),]
            obs_, reward, done, _, _ = env.step(action)
            obs_ = obs_ / 255.0
            obs_ = torch.from_numpy(obs_).float().unsqueeze(0)
            state_ = encoder(obs_)
            state = state_
            sum_reward += reward
            if done:
                print(f"{j + 1} Evaluation reward: {sum_reward}")
                eval_sum_reward += sum_reward
                break
    print(f"Average Evaluation reward: {eval_sum_reward / rounds}")


if __name__ == '__main__':
    eval(
        20,
        encoder_path="/home/wsl/pythonWork/paper_one/log/encoder1/cheakpoint/obs_encoder.pth",
        policy_path="/home/wsl/pythonWork/paper_one/log/trainer1/best_save_dir/policy_net.pth"
    )
        
