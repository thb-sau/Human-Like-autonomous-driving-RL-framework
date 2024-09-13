import os
import h5py
from pathlib import Path
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
ROOT_PATH = Path(__file__).parents[0].absolute()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class ReplayBuffer(Dataset):
    def __init__(self, buffer_size, state_dim, action_dim):
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.rewards_c = np.zeros((buffer_size, 1))
        self.next_states = np.zeros((buffer_size, state_dim))
        self.done = np.zeros((buffer_size, 1))
        self.td_error = np.zeros((buffer_size, 1))

        self.buffer_size = buffer_size
        self.now_buffer_size = 0

        self.p = 0
        
            
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        sample = {'states': self.states[idx], 'actions': self.actions[idx], 'rewards': self.rewards[idx], 'next_states': self.next_states[idx]}
        return sample
    
    def add_per(self, state, action, reward, reward_c, next_state, done):
        
        max_td_error = np.max(self.td_error)
        if self.now_buffer_size < self.buffer_size:
            self.states[self.now_buffer_size] = state
            self.actions[self.now_buffer_size] = action
            self.rewards[self.now_buffer_size] = reward
            self.rewards_c[self.now_buffer_size] = reward_c
            self.next_states[self.now_buffer_size] = next_state
            self.done[self.now_buffer_size] = 1 if done else 0
            self.td_error[self.now_buffer_size] = max_td_error
            self.now_buffer_size += 1
        else:
            # 找到td_error最小的index
            pointer = np.argmin(self.td_error)
            self.states[pointer] = state
            self.actions[pointer] = action
            self.rewards[pointer] = reward
            self.rewards_c[pointer] = reward_c
            self.next_states[pointer] = next_state
            self.done[pointer] = 1 if done else 0
            self.td_error[pointer] = max_td_error

    
    def add(self, state, action, reward, reward_c, next_state, done):
        
        max_td_error = np.max(self.td_error)
        if self.now_buffer_size < self.buffer_size:
            self.states[self.now_buffer_size] = state
            self.actions[self.now_buffer_size] = action
            self.rewards[self.now_buffer_size] = reward
            self.rewards_c[self.now_buffer_size] = reward_c
            self.next_states[self.now_buffer_size] = next_state
            self.done[self.now_buffer_size] = 1 if done else 0
            self.td_error[self.now_buffer_size] = max_td_error
            self.now_buffer_size += 1
        else:
            pointer = self.p % self.buffer_size
            self.states[pointer] = state
            self.actions[pointer] = action
            self.rewards[pointer] = reward
            self.rewards_c[pointer] = reward_c
            self.next_states[pointer] = next_state
            self.done[pointer] = 1 if done else 0
            self.td_error[pointer] = max_td_error
            self.p += 1
        

    def sample_per(self, batch_size):
        p = np.squeeze(softmax(self.td_error[0:self.now_buffer_size]))
        assert np.isclose(p.sum(), 1.0), "Probabilities do not sum up to 1"
        index = np.random.choice(self.now_buffer_size, size=batch_size, p=p, replace=False)
        bn_s, bn_a, bn_r, bn_r_c, bn_s_, bn_d = self.states[index], self.actions[index], self.rewards[index],\
                                        self.rewards_c[index], self.next_states[index], self.done[index]

        return bn_s, bn_a, bn_r, bn_r_c, bn_s_, bn_d, index
    
    def sample(self, batch_size):
        index = np.random.choice(self.now_buffer_size, size=batch_size, replace=False)
        bn_s, bn_a, bn_r, bn_r_c, bn_s_, bn_d = self.states[index], self.actions[index], self.rewards[index],\
                                        self.rewards_c[index], self.next_states[index], self.done[index]

        return bn_s, bn_a, bn_r, bn_r_c, bn_s_, bn_d, index
    
    def update_td_error(self, td_error, index):
        self.td_error[index] = td_error

    def get_p(self, index):
        return softmax(self.td_error[0:self.now_buffer_size])[index]
    

class CNNReplayBuffer(Dataset):
    def __init__(self, buffer_size, obs_shape, action_dim):
        self.obs = np.zeros((buffer_size, obs_shape[0], obs_shape[1], obs_shape[2]))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.rewards_c = np.zeros((buffer_size, 1))
        self.next_obs = np.zeros((buffer_size, obs_shape[0], obs_shape[1], obs_shape[2]))
        self.done = np.zeros((buffer_size, 1))

        self.buffer_size = buffer_size
        self.now_buffer_size = 0

        self.p = 0
        
            
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        sample = {'states': self.obs[idx], 'actions': self.actions[idx], 'rewards': self.rewards[idx], 'next_states': self.next_obs[idx]}
        return sample
    

    
    def add(self, obs, action, reward, reward_c, next_obs, done):
        
        if self.now_buffer_size < self.buffer_size:
            self.obs[self.now_buffer_size] = obs
            self.actions[self.now_buffer_size] = action
            self.rewards[self.now_buffer_size] = reward
            self.rewards_c[self.now_buffer_size] = reward_c
            self.next_obs[self.now_buffer_size] = next_obs
            self.done[self.now_buffer_size] = 1 if done else 0
            self.now_buffer_size += 1
        else:
            pointer = self.p % self.buffer_size
            self.obs[pointer] = obs
            self.actions[pointer] = action
            self.rewards[pointer] = reward
            self.rewards_c[pointer] = reward_c
            self.next_obs[pointer] = next_obs
            self.done[pointer] = 1 if done else 0
            self.p += 1
        

    
    def sample(self, batch_size):
        index = np.random.choice(self.now_buffer_size, size=batch_size, replace=False)
        bn_s, bn_a, bn_r, bn_r_c, bn_s_, bn_d = self.obs[index], self.actions[index], self.rewards[index],\
                                        self.rewards_c[index], self.next_obs[index], self.done[index]

        return bn_s, bn_a, bn_r, bn_r_c, bn_s_, bn_d, index
    
