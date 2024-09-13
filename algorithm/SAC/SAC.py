import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym

from network.net import Actor, Critic, Q, Discriminator, PredictNextState
from network.encoder import SimpleViTEncoder
from data.expert_dataset import EncodeExpertDataset
from data.replay_buffer import ReplayBuffer

from config.config import load_config
encoder_config = load_config('encoder.yaml', 'encoder')

class SAC(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_space: gym.spaces.Box, 
                 encoder_model_path, 
                 buffer_size, 
                 batch_size, 
                 lr, 
                 gamma, 
                 tau, # target value net update rate
                 log_dir, 
                 gradient_steps,
                 device='cuda',
                 replay_mode='default',  # or per
                 ) -> None:
        super(SAC, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.replay_mode = replay_mode

        action_dim = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high

        self.target_entropy = -action_dim

        self.policy_net = Actor(state_dim, action_space.shape[0]).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        # 一个shape为(batch_size, 1)可更新的tensor
        alpha = torch.ones(batch_size, 1, requires_grad=True).float().to(device) * 0.01
        log_alpha = torch.log(alpha)
        self.log_alpha_tensor = nn.Parameter(log_alpha)
        

        def weights_he_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                # 使用Kaiming初始化，默认为ReLU激活函数
                init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    # 偏置项通常初始化为0
                    init.constant_(m.bias.data, 0)

        # 应用HE初始化到你的网络
        self.policy_net.apply(weights_he_init)
        self.value_net.apply(weights_he_init)
        self.Target_value_net.apply(weights_he_init)
        self.Q_net1.apply(weights_he_init)
        self.Q_net2.apply(weights_he_init)


        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=lr)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha_tensor], lr=lr)
        
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
        

        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        log_dir = Path(log_dir)
        self.writer = SummaryWriter(str(log_dir / 'tensorboard'))
        self.num_training = 0
        self.steps = 0
    

    def select_action(self, obs):
        with torch.no_grad():
            # bug 会变成NaN
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = self.encoder(obs)
            # 如果state中有NaN值触发异常
            if torch.isnan(state).any():
                raise ValueError("state has NaN values")
            mu, log_std = self.policy_net(state)
            sigma = torch.exp(log_std)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z).squeeze(0).detach().cpu().numpy()
            return action
    

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = batch_mu + batch_sigma*z.to(self.device)
        log_prob = dist.log_prob(action)
        return torch.tanh(action), log_prob.sum(dim=-1).unsqueeze(1), z, batch_mu, batch_log_sigma
    
    def value_loss(self, bn_s):
        # !!!Note that the actions are sampled according to the current policy,
        # instead of replay buffer. (From original paper)
        bn_s = torch.tensor(bn_s, dtype=torch.float32).to(self.device)
        sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
        excepted_value = self.value_net(bn_s)
        excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
        V_loss = self.value_criterion(excepted_value, excepted_new_Q.detach()).mean()  # J_V
        return V_loss
    
    def dual_q_loss(self, bn_s, bn_a, bn_r, bn_s_, bn_d, index):
        bn_s = torch.tensor(bn_s, dtype=torch.float32).to(self.device)
        bn_a = torch.tensor(bn_a, dtype=torch.float32).to(self.device)
        bn_r = torch.tensor(bn_r, dtype=torch.float32).to(self.device)
        bn_s_ = torch.tensor(bn_s_, dtype=torch.float32).to(self.device)
        bn_d = torch.tensor(bn_d, dtype=torch.float32).to(self.device)
        excepted_Q1 = self.Q_net1(bn_s, bn_a)
        excepted_Q2 = self.Q_net2(bn_s, bn_a)
        target_value = self.Target_value_net(bn_s_)
        next_q_value = bn_r + (1 - bn_d) * self.gamma * target_value
        Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean() # J_Q
        Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()
        if self.replay_mode == 'per':
            td_error = torch.abs(torch.min(excepted_Q1.clone().detach(), excepted_Q2.clone().detach()) - next_q_value.clone().detach()).clone().detach()
            self.replay_buffer.update_td_error(td_error.cpu().numpy(), index)
        return Q1_loss, Q2_loss
    

    def policy_loss(self, bn_s):
        bn_s = torch.tensor(bn_s, dtype=torch.float32).to(self.device)
        sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
        
        excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
        
        alpha = torch.exp(self.log_alpha_tensor)
        pi_loss = (alpha.detach() * log_prob - excepted_new_Q).mean() # according to original paper

        alpha = torch.exp(self.log_alpha_tensor)
        log_alpha_loss = - (alpha * (log_prob + self.target_entropy).detach()).mean()
        
        return pi_loss, log_alpha_loss, log_prob.clone().mean().detach()
       
    
    
    def add(self, obs, action, r_e, r_c, next_obs, done):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        state = self.encoder(obs).cpu().squeeze(0).detach().numpy()
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = self.encoder(next_obs).cpu().squeeze(0).detach().numpy()
        if self.replay_mode == 'per':
            self.replay_buffer.add_per(state, action, r_e, r_c, next_state, done)
        else:
            self.replay_buffer.add(state, action, r_e, r_c, next_state, done)

    def update(self):
        self.steps += 1
        if self.replay_buffer.now_buffer_size < self.batch_size:
            return

        for _ in range(self.gradient_steps):
            bn_s, bn_a, bn_r, _, bn_s_, bn_d, index = self.replay_buffer.sample(self.batch_size)

            # 更新v
            V_loss = self.value_loss(bn_s)
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            # 更新Q
            # Dual Q net
            Q1_loss, Q2_loss = self.dual_q_loss(bn_s, bn_a, bn_r, bn_s_, bn_d, index) 
            
            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

            
            # 更新策略网络
            pi_loss, log_alpha_loss, entropy = self.policy_loss(bn_s)
            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.log_alpha_tensor, 0.5)
            self.log_alpha_optimizer.step()
            

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/log_alpha_loss', log_alpha_loss, global_step=self.num_training)
            self.writer.add_scalar('entropy', entropy, global_step=self.num_training)
            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

            self.num_training += 1


    def save(self, path: Path):
        torch.save(self.policy_net.state_dict(), str(path/ 'policy_net.pth'))
        torch.save(self.value_net.state_dict(), str(path / 'value_net.pth'))
        torch.save(self.Q_net1.state_dict(), str(path / 'Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), str(path / 'Q_net2.pth'))
        torch.save(self.log_alpha_tensor, str(path / 'log_alpha_tensor.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, path: Path):
        self.policy_net.load_state_dict(torch.load(str(path / 'policy_net.pth')))
        self.value_net.load_state_dict(torch.load(str(path / 'value_net.pth')))
        self.Q_net1.load_state_dict(torch.load(str(path / 'Q_net1.pth')))
        self.Q_net2.load_state_dict(torch.load(str(path / 'Q_net2.pth')))
        self.log_alpha_tensor = torch.load(str(path / 'log_alpha_tensor.pth'))
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        print("====================================")
        print("model has been loaded...")
        print("====================================")