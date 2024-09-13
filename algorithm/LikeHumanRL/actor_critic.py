import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym

from network.net import Actor, Critic, Q, Discriminator, PredictNextState, DiscriminatorSCA
from network.encoder import SimpleViTEncoder
from data.expert_dataset import EncodeExpertDataset
from data.replay_buffer import ReplayBuffer
from utils import normalization

from config.config import load_config
encoder_config = load_config('encoder.yaml', 'encoder')


class ActorCritic(nn.Module):
    def __init__(self, 
                 env_mode,
                 state_dim, 
                 action_space: gym.spaces.Box, 
                 expert_data_name, 
                 encoder_model_path, 
                 buffer_size, 
                 batch_size, 
                 lr, 
                 gamma, 
                 tau, # target value net update rate
                 log_dir, 
                 gradient_steps,
                 alpha,
                 device='cuda',
                 discriminator_mode = 'SCA',
                 replay_mode = 'default',  # or per
                 update_mode = 'value'  #  or policy
                 ) -> None:
        super(ActorCritic, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.discriminator_mode = discriminator_mode
        self.replay_mode = replay_mode
        self.alpha = alpha
        self.update_mode = update_mode

        action_dim = action_space.shape[0]
        self.action_low = action_space.low
        self.action_high = action_space.high

        self.target_entropy = -action_dim

        self.policy_net = Actor(state_dim, action_space.shape[0]).to(device)
        self.value_net = Critic(state_dim).to(device)
        self.Target_value_net = Critic(state_dim).to(device)
        self.Q_net1 = Q(state_dim, action_dim).to(device)
        self.Q_net2 = Q(state_dim, action_dim).to(device)

        if self.discriminator_mode == 'SCA':
            self.discriminator = DiscriminatorSCA(state_dim, action_dim).to(device)
        else:
            self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.log_alpha_tensor = nn.Parameter(torch.randn(batch_size, 1, requires_grad=True).float().to(device))

        def weights_he_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

        self.policy_net.apply(weights_he_init)
        self.value_net.apply(weights_he_init)
        self.Target_value_net.apply(weights_he_init)
        self.Q_net1.apply(weights_he_init)
        self.Q_net2.apply(weights_he_init)
        self.discriminator.apply(weights_he_init)


        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()
        self.next_state_criterion = nn.MSELoss()

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)
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
        ).to(self.device)

        try:
            # Attempt to load the model
            print("Loading model...")
            self.encoder.load_state_dict(torch.load(encoder_model_path, map_location=torch.device(self.device)))
            self.encoder.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
        

        self.expert_dataset = EncodeExpertDataset(env_mode, expert_data_name, encoder_model_path)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        log_dir = Path(log_dir)
        self.writer = SummaryWriter(str(log_dir / 'tensorboard'))
        self.num_training = 0

        self.steps = 0


    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            state = self.encoder(obs)
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
        # dist.entropy
        return torch.tanh(action), log_prob.sum(dim=1).unsqueeze(1), z, batch_mu, batch_log_sigma
    
    def discriminator_InfoNCE_loss(self):
        """
        InfoNCE损失
        """
        # 正样本对
        with torch.no_grad():
            states, r_e, r_c, actions = self.expert_dataset.sample(self.batch_size)
            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            if self.discriminator_mode == 'SCA':
                r_e = torch.tensor(r_e, dtype=torch.float32).to(self.device).unsqueeze(1)
                r_c = torch.tensor(r_c, dtype=torch.float32).to(self.device).unsqueeze(1)
            actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
            
            neg_num = 2
            s = states.repeat(neg_num, 1)
            neg_sample_actions, _, _, _, _ = self.evaluate(s)
            # nag_random_actions = (torch.rand(self.batch_size * 2, 2) * 2 - 1).to(self.device)
            # nag_actions = torch.cat([neg_sample_actions, nag_random_actions], dim=0) 
            
        if self.discriminator_mode == 'SCA':
            neg_scores = self.discriminator(s, r_e.repeat(neg_num, 1), r_c.repeat(neg_num, 1), neg_sample_actions).reshape(-1, neg_num)  # (batch_size, 10)
            pos_scores = self.discriminator(states, r_e, r_c, actions)
        else: 
            neg_scores = self.discriminator(s, neg_sample_actions).reshape(-1, neg_num)  # (batch_size, 10)
            pos_scores = self.discriminator(states, actions) 
        
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        
        probabilities = F.softmax(all_scores, dim=1) 
        
        pos_prob = probabilities[:, 0]
        
        loss = -torch.log(pos_prob).mean()  
        return loss
    
    
    def value_loss(self, bn_s, r_e, r_c):
        # !!!Note that the actions are sampled according to the current policy,
        # instead of replay buffer. (From original paper)
        bn_s = torch.tensor(bn_s, dtype=torch.float32).to(self.device)
        sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
        excepted_value = self.value_net(bn_s)
        if self.update_mode == 'policy':
            excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
            V_loss = self.value_criterion(excepted_value, excepted_new_Q.detach()).mean()  # J_V
            return V_loss
        if self.discriminator_mode == 'SCA':
            r_e = torch.tensor(r_e, dtype=torch.float32).to(self.device)
            r_c = torch.tensor(r_c, dtype=torch.float32).to(self.device)
            discriminate = self.discriminator(bn_s, r_e, r_c, sample_action)
        else:
            discriminate = self.discriminator(bn_s, sample_action)
        excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action)) + self.alpha * discriminate
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
        Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()) # J_Q
        Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach())
        td_error = torch.abs(torch.min(excepted_Q1.clone().detach(), excepted_Q2.clone().detach()) - next_q_value.clone().detach()).clone().detach()

        if self.replay_mode == 'per':
            #  lr = lr * (n * p_t) ** -beat
            # n = self.replay_buffer.now_buffer_size 
            # p  = self.replay_buffer.get_p(index)
            # a = (n * p) ** (-0.4)
            # a = torch.tensor(a).float().to(self.device)
            # Q1_loss = a * Q1_loss
            # Q2_loss = a * Q2_loss
            self.replay_buffer.update_td_error(td_error.cpu().numpy(), index)

        return Q1_loss.mean(), Q2_loss.mean()
    
    
    def policy_loss(self, bn_s, bn_r, bn_r_c):
        """
        return:
            policy_loss: policy loss
            actor_loss: actor loss
            log_prob: log probability of action
            discriminate: discriminator output
            entropy_lambda_loss: entropy lambda loss
            discriminator_lambda_loss: discriminator lambda loss
        """
        bn_s = torch.tensor(bn_s, dtype=torch.float32).to(self.device)
        sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)
        if self.discriminator_mode == 'SCA':
            r_e = torch.tensor(bn_r, dtype=torch.float32).to(self.device)
            r_c = torch.tensor(bn_r_c, dtype=torch.float32).to(self.device)
            discriminate = self.discriminator(bn_s, r_e, r_c, sample_action)
        else:
            discriminate = self.discriminator(bn_s, sample_action)
        excepted_new_Q = torch.min(self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action))
        if self.update_mode == 'policy':
            alpha = torch.exp(self.log_alpha_tensor)
            pi_loss = ((0.5 - discriminate) * alpha.detach() - excepted_new_Q).mean()
            log_alpha_loss = (-1 * ((0.5 - discriminate).clone().detach() * alpha)).mean()
            actor_loss = (-1 * excepted_new_Q.clone()).mean().detach().squeeze()
            return pi_loss, actor_loss, log_prob.clone().mean().detach().squeeze(), discriminate.clone().mean().detach().squeeze(), log_alpha_loss
        
        pi_loss = (- excepted_new_Q - self.alpha * discriminate).mean()
        actor_loss = (-1 * excepted_new_Q.clone()).mean().detach().squeeze()
        
        return pi_loss, actor_loss, log_prob.clone().mean().detach().squeeze(), discriminate.clone().mean().detach().squeeze()


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
            if self.replay_mode == 'per':
                bn_s, bn_a, bn_r_e, bn_r_c, bn_s_, bn_d, index = self.replay_buffer.sample_per(self.batch_size)
            else:
                bn_s, bn_a, bn_r_e, bn_r_c, bn_s_, bn_d, index = self.replay_buffer.sample(self.batch_size)

            bn_r = np.where(bn_r_e >= 1, 1.0, np.where(bn_r_e < 0, -1.0, 0.0))

        
            discriminator_loss = self.discriminator_InfoNCE_loss()
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
            self.discriminator_optimizer.step()
            # print(discriminator_loss)

           
            V_loss = self.value_loss(bn_s, bn_r_e, bn_r_c)
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

           
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

            
            if self.update_mode == 'policy':
                pi_loss, actor_loss, log_prob, discriminate, log_alpha_loss = self.policy_loss(bn_s, bn_r_e, bn_r_c)
                self.policy_optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

                self.log_alpha_optimizer.zero_grad()
                log_alpha_loss.backward()
                nn.utils.clip_grad_norm_(self.log_alpha_tensor, 0.5)
                self.log_alpha_optimizer.step()
                self.writer.add_scalar('Loss/log_alpha_loss', log_alpha_loss, global_step=self.num_training)
            else:
                pi_loss, actor_loss, log_prob, discriminate = self.policy_loss(bn_s, bn_r_e, bn_r_c)
                self.policy_optimizer.zero_grad()
                pi_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()
            

            self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q1_loss', Q1_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/Q2_loss', Q2_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/discriminator_loss', discriminator_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_training)
            self.writer.add_scalar('/discriminate', discriminate, global_step=self.num_training)
            self.writer.add_scalar('/log_prob', log_prob, global_step=self.num_training)
            self.writer.flush()
            # update target v net update
            for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(target_param * (1 - self.tau) + param * self.tau)

            self.num_training += 1

    def save(self, path: Path):
        torch.save(self.policy_net.state_dict(), str(path/ 'policy_net.pth'))
        torch.save(self.value_net.state_dict(), str(path / 'value_net.pth'))
        torch.save(self.Q_net1.state_dict(), str(path / 'Q_net1.pth'))
        torch.save(self.Q_net2.state_dict(), str(path / 'Q_net2.pth'))
        torch.save(self.discriminator.state_dict(), str(path / 'discriminator.pth'))
        torch.save(self.log_alpha_tensor, str(path / 'log_alpha.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, path: Path):
        self.policy_net.load_state_dict(torch.load(str(path / 'policy_net.pth')))
        self.value_net.load_state_dict(torch.load(str(path / 'value_net.pth')))
        self.Q_net1.load_state_dict(torch.load(str(path / 'Q_net1.pth')))
        self.Q_net2.load_state_dict(torch.load(str(path / 'Q_net2.pth')))
        self.discriminator.load_state_dict(torch.load(str(path / 'discriminator.pth')))
        self.log_alpha_tensor = torch.load(str(path / 'log_alpha.pth'))
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        print("====================================")
        print("model has been loaded...")
        print("====================================")