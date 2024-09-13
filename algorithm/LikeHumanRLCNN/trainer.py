import sys
import os
import warnings
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch

ROOT_PATH = Path(__file__).parents[2].absolute()
sys.path.append(str(ROOT_PATH))

from algorithm.LikeHumanRLCNN.actor_critic import ActorCritic
from smarts_env.env import make_env
from utils import QueueAsStack, SuccessRateWriter

from config.config import load_config
config = load_config('config.yaml', 'LHRL')


# warnings.filterwarnings("ignore", message="vehicle id named .* is more than 50 characters long.*")
warnings.filterwarnings("ignore")

# 固定随机种子
np.random.seed(config.seed)
torch.manual_seed(config.seed)

def trainer(log_dir: str, env_mode: str, expert_data_name,buffer_size=10000, update_mode='value'):

    train_env = make_env(env_mode, config.max_episode_steps, config.seed, config.train_headless, config.reward_mode)
    eval_env = make_env(env_mode, config.max_episode_steps, config.seed, config.eval_headless, config.reward_mode)

    model = ActorCritic(
        env_mode=env_mode, 
        action_space=train_env.action_space,
        expert_data_name=expert_data_name,
        buffer_size=buffer_size,
        batch_size=config.batch_size,
        lr=config.lr,
        gamma=config.gamma,
        tau=config.tau,
        alpha=config.alpha,
        log_dir=log_dir,
        gradient_steps=config.gradient_steps,
        update_mode=update_mode
    )

    steps = config.steps
    eval_freq = config.eval_freq
    eval_episode = config.eval_episode
    save_interval = config.save_interval

    log_dir = Path(log_dir)
    save_dir = log_dir / "model_trained"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_save_dir = log_dir / "best_save_dir"
    best_save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir / 'tensorboard'))
    best_reward = -float('inf')

    success_writer = SuccessRateWriter(100, str(log_dir))

    length_stack = QueueAsStack(max_size=20)
    reward_stack = QueueAsStack(max_size=20)
    success_stack = QueueAsStack(max_size=20)
    step_num = 0
    epoch = 0
    while True:
        train_ep_length = 0
        train_ep_sum_reward = 0
        obs, _ = train_env.reset()
        obs = obs / 255.0
        while True:
            action = model.select_action(obs)
            obs_, reward, done, _, info = train_env.step(action)
            obs_ = obs_ / 255.0
            model.add(obs, action, reward, info['neighborhood_reward'], obs_, done)
            model.update()
            obs = obs_
            train_ep_sum_reward += reward
            train_ep_length += 1
            step_num += 1
            if done:
                length_stack.push(train_ep_length)
                reward_stack.push(train_ep_sum_reward)
                if info['reached_goal']:
                    success_stack.push(1.0)
                else:
                    success_stack.push(0.0)
                break
        epoch += 1
        writer.add_scalar('train/train_reward_mean', reward_stack.mean(), epoch)
        writer.add_scalar('train/ep_length_mean', length_stack.mean(), epoch)
        success_rate = success_stack.mean()
        writer.add_scalar('train/success_rate', success_rate, epoch)
        if success_writer.write_data(success_rate, step_num):
            success_writer.write_matrix_to_file()
            success_writer.clear_matrix()

        # eval
        if epoch % eval_freq == 0:
            print("="*20, "Evaluating", "="*20)
            eval_sum_reward = 0
            eval_length = 0
            eval_success = 0
            for j in range(eval_episode):
                eval_ep_sum_reward = 0
                obs, _ = eval_env.reset()
                obs = obs / 255.0
                while True:
                    eval_length += 1
                    action = model.select_action(obs)
                    obs_, reward, done, _, info = eval_env.step(action)
                    obs_ = obs_ / 255.0
                    obs = obs_
                    eval_ep_sum_reward += reward
                    if done:
                        print(f"{j + 1} Evaluation reward: {eval_ep_sum_reward}")
                        eval_sum_reward += eval_ep_sum_reward
                        if info['reached_goal']:
                            eval_success += 1.0
                        break
            writer.add_scalar('eval/eval_reward_mean', eval_sum_reward / eval_episode, epoch)
            writer.add_scalar('eval/ep_length_mean', eval_length / eval_episode, epoch)
            writer.add_scalar('eval/success_rate', eval_success / eval_episode, epoch)
            
            if eval_sum_reward / eval_episode > best_reward:
                print("Best reward updated!")
                model.save(best_save_dir)
                best_reward = eval_sum_reward / eval_episode
        writer.flush()
        # save
        if epoch % save_interval == 0:
            model.save(save_dir)

        if step_num >= steps:
            break

        
    
    print("Training Done!")
    model.save(save_dir)
    success_writer.write_matrix_to_file()
    success_writer.clear_matrix()

if __name__ == '__main__':
    trainer(log_dir="log/cnn_radical_left_t2/", env_mode="left_t", update_mode='value', expert_data_name="")
    
        
