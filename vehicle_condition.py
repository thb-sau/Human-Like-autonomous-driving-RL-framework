import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils import select_action, write_array_to_txt
from smarts_env.env import make_env
from network.encoder import SimpleViTEncoder
from network.net import Actor
from config.config import load_config
encoder_config = load_config('encoder.yaml', 'encoder')

warnings.filterwarnings("ignore")


def evaluate(encoder_model_path, actor_model_path, env, log_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SimpleViTEncoder(
        image_size=80,
        channels=9,
        patch_size=encoder_config.patch_size,
        dim=encoder_config.dim,
        depth=encoder_config.n_layer,
        heads=encoder_config.n_head,
        mlp_dim=encoder_config.mlp_dim,
        dim_head=encoder_config.dim_head,
    ).to(device)

    try:
        # 尝试加载模型
        print("正在加载encoder...")
        encoder.load_state_dict(torch.load(encoder_model_path, map_location=torch.device(device)))
        encoder.eval()
        print("encoder加载成功。")
    except Exception as e:
        print(f"encoder加载时发生错误：{e}")

    actor =  Actor(
        state_dim=encoder_config.dim,
        max_action=env.action_space.shape[0]
    ).to(device)

    try:
        # 尝试加载模型
        print("正在加载actor...")
        actor.load_state_dict(torch.load(actor_model_path, map_location=torch.device(device)))
        actor.eval()
        print("actor加载成功。")
    except Exception as e:
        print(f"actor加载时发生错误：{e}")

    log_dir = Path(log_dir)
    writer = SummaryWriter(str(log_dir / 'vehicle_condition'))

    max_step = 400
    ep = 100
    e = 0
    i = 0

    sm = np.zeros(shape=(ep, max_step), dtype=np.float32)
    am = np.zeros(shape=(ep, max_step), dtype=np.float32)

    while e < ep:
        sm_t = np.zeros(shape=(max_step,), dtype=np.float32)
        am_t = np.zeros(shape=(max_step,), dtype=np.float32)
        eval_ep_sum_reward = 0
        step = 0
        obs, _ = env.reset()
        obs = obs / 255.0
        while True:
            action = select_action(encoder, actor, obs, device)
            # print(action)
            obs_, reward, done, _, eval_info = env.step(action)
            obs_ = obs_ / 255.0
            obs = obs_
            eval_ep_sum_reward += reward
            speed = env.get_ego_now_speed()
            acceleration = env.get_ego_now_acceleration()
            # 如果acceleration是nan
            if np.isnan(acceleration):
                acceleration = 0
            sm_t[step] = speed
            am_t[step] = acceleration
            yaw_rate = env.get_ego_now_yaw_rate()
            writer.add_scalar(f'{i + 1}ep/speed', speed, global_step=step)
            writer.add_scalar(f'{i + 1}ep/acceleration', acceleration, global_step=step)
            writer.add_scalar(f'{i + 1}ep/yaw_rate', yaw_rate, global_step=step)
            step += 1
            if done:
                print(f"{i + 1} Evaluation reward: {eval_ep_sum_reward}")
                break
        i += 1
        if eval_info['reached_goal']:
            sm[e] = sm_t
            am[e] = am_t
            e += 1
            if e == ep:
                break 
    asm = sm.mean(axis=0)
    aam = am.mean(axis=0)
    write_array_to_txt(asm, str(log_dir / 'speed.txt'))
    write_array_to_txt(aam, str(log_dir / 'acceleration.txt'))
              


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_mode', type=str, default='left_t')
    parser.add_argument('--encoder_model_path', type=str, default='')
    parser.add_argument('--policy_mode_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    args = parser.parse_args()

    env = make_env(args.env_mode, 400, 233, False)

    evaluate(args.encoder_model_path, args.policy_mode_dir, env, args.log_dir)
    
