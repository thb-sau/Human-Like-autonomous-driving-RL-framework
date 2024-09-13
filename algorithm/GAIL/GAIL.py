import os
from pathlib import Path
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import gymnasium as gym

from network.net import Actor, Critic, Discriminator
from network.encoder import SimpleViTEncoder
from data.expert_dataset import EncodeExpertDataset

from config.config import load_config
encoder_config = load_config('encoder.yaml', 'encoder')


Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])


class GAIL(nn.Module):
    def __init__(self,
                 state_dim, 
                 action_space: gym.spaces.Box, 
                 encoder_model_path, 
                 buffer_size, 
                 batch_size, 
                 lr, 
                 gamma, 
                 clip_param,
                 max_grad_norm,
                 log_dir, 
                 gradient_steps,
                 env_mode,
                 expert_data_name,
                 device='cuda' 
                ):
        super(GAIL, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gradient_steps = gradient_steps

        action_dim = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high
        self.action_dim = action_dim
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm

        assert os.path.exists(encoder_model_path), "File does not exist, please verify the path is correct."
        assert encoder_model_path.endswith('.pth'), "File is not in .pth format, please provide a .pth file."

        self.encoder = SimpleViTEncoder(
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
            # Attempt to load the model
            print("Loading model...")
            self.encoder.load_state_dict(torch.load(encoder_model_path, map_location=torch.device(device)))
            self.encoder.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

        self.expert_dataset = EncodeExpertDataset(env_mode, expert_data_name, encoder_model_path)

        self.actor_net = Actor(state_dim, action_space.shape[0]).to(device)
        self.critic_net = Critic(state_dim).to(device)
        self.discriminator = Discriminator(state_dim, action_dim).to(device)

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.num_training = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr)
        
        log_dir = Path(log_dir)
        self.writer = SummaryWriter(str(log_dir / 'tensorboard'))

        def weights_he_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                # Use Kaiming initialization, default for ReLU activation function
                init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    # Bias terms are usually initialized to 0
                    init.constant_(m.bias.data, 0)

        # Apply HE initialization to your networks
        self.actor_net.apply(weights_he_init)
        self.critic_net.apply(weights_he_init)
        self.discriminator.apply(weights_he_init)
        self.steps = 0
       

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = self.encoder(obs)
            if torch.isnan(state).any():
                raise ValueError("state has NaN values")
            mu, log_std = self.actor_net(state)
            sigma = torch.exp(log_std)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z).squeeze().detach().cpu().numpy()
            action_log_prob = dist.log_prob(z).sum(dim=-1)
            return action, action_log_prob

    def get_discriminator_reward(self, obs, action):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = self.encoder(obs)
            action = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
            reward = self.discriminator(state, action).cpu().squeeze(0).detach().numpy()
            return reward[0]


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def add(self, obs, action, reward, action_log_prob, next_obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        state = self.encoder(obs).cpu().squeeze(0).detach().numpy()
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = self.encoder(next_obs).cpu().squeeze(0).detach().numpy()
        transition = Transition(state, action, reward, action_log_prob, next_state)
        self.buffer.append(transition)
        self.counter+=1
        if self.counter % self.buffer_size == 0:
            self.counter = 0
            return True
        return False
    
    def discriminator_loss(self, neg_states, neg_actions):
        # Positive sample pairs
        states, _, _, actions = self.expert_dataset.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        # Generate a random number with shape (batch_size, 2) within the interval [0, 1)
        uniform_noise = torch.rand(self.batch_size, actions.shape[-1])
        # Linearly transform the random numbers within [0, 1) to the interval [-0.1, 0.1]
        noise = -0.1 + uniform_noise * 0.2
        # Add the replicated tensor with the bounded random noise
        actions = torch.clip(actions + noise.to(self.device), -1, 1)
        pos = (states.clone(), actions.clone())

        # Negative samples
        neg = (neg_states, neg_actions)

        pos_scores = self.discriminator(*pos)
        pos_labels = torch.ones_like(pos_scores)
        neg_scores = self.discriminator(*neg)
        neg_labels = torch.zeros_like(neg_scores)
        
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def update(self):
        state = torch.tensor([t.state for t in self.buffer ], dtype=torch.float).to(self.device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, self.action_dim).to(self.device)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).to(self.device)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.device)

        
        with torch.no_grad():
            target_v = reward + self.gamma * self.critic_net(next_state)

        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(self.gradient_steps): # iteration ppo_epoch 
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_size)), self.batch_size, drop_last=True):
                discriminator_loss = self.discriminator_loss(state[index], action[index])
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.discriminator_optimizer.step()

                # epoch iteration, PPO core!!!
                mu, log_sigma = self.actor_net(state[index])
                sigma = torch.exp(log_sigma)
                n = Normal(mu, sigma)
                action_log_prob = n.log_prob(action[index]).sum(dim=-1).unsqueeze(1)
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                
                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage[index]
                pi_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                self.num_training += 1

                self.writer.add_scalar('loss/discriminator_loss', discriminator_loss, self.num_training)
                self.writer.add_scalar('loss/value_loss', value_loss, self.num_training)
                self.writer.add_scalar('loss/pi_loss', pi_loss, self.num_training)
            self.training_step += 1

        del self.buffer[:]
        self.steps += 1


    def save(self, path: Path):
        torch.save(self.actor_net.state_dict(), str(path/ 'actor_net.pth'))
        torch.save(self.critic_net.state_dict(), str(path / 'critic_net.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, path: Path):
        self.actor_net.load_state_dict(torch.load(str(path / 'actor_net.pth')))
        self.critic_net.load_state_dict(torch.load(str(path / 'critic_net.pth')))
        print("====================================")
        print("model has been loaded...")
        print("====================================")