import os
import h5py
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
ROOT_PATH = Path(__file__).parents[0].absolute()

from network.encoder import SimpleViTEncoder

from config.config import load_config
config = load_config('encoder.yaml', 'encoder')

class ExpertDataset(Dataset):
    def __init__(self):
        super(ExpertDataset, self).__init__()
        self.obs = []
        self.actions = []
        self.rewards = []
        self.rewards_ = []
        self.lenth=0
    
    def loade_dataset(self, data_name):
        file_path = str(ROOT_PATH / data_name)
        
        with h5py.File(file_path, 'r') as hf:
            lenth = hf['lenth'][()]
            self.lenth += lenth
            for i in range(lenth):
                trajectory = hf[f'trajectory_{i}']
                obs = trajectory['obs'][()]
                obs = obs / 255.0
                actions = trajectory['actions'][()]
                r_e = trajectory['rewards'][()]
                r_c = trajectory['rewards_'][()]
                
                self.obs.extend(obs)
                self.actions.extend(actions)
                self.rewards.extend(r_e)
                self.rewards_.extend(r_c)

    def start(self):
        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.rewards_ = np.array(self.rewards_)
        
        print("数据加载完成 轨迹数:", self.lenth, "数据量:", self.obs.shape[0])


    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        sample = {'obs': self.obs[idx], 'actions': self.actions[idx], 'rewards': self.rewards[idx]}
        return sample
    
    def sample(self, batch_size):
        index = np.random.choice(range(self.__len__()), batch_size, replace=False)
        bn_obs, bn_r_e, bn_r_c, bn_a = self.obs[index], self.rewards[index], self.rewards_[index], self.actions[index]
        return  bn_obs, bn_r_e, bn_r_c, bn_a
    

class EncodeExpertDataset(Dataset):
    def __init__(self,env_mode, data_name, model_path):
        super(EncodeExpertDataset, self).__init__()
        self.file_path = str(ROOT_PATH / data_name)
        self.states = []
        self.actions = []
        self.r_e = []
        self.r_c = []

        assert os.path.exists(model_path), "File does not exist, please verify the path is correct."
        assert model_path.endswith('.pth'), "File is not in .pth format, please provide a .pth file."
        
        encoder = SimpleViTEncoder(
            image_size=80,
            channels=9,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.n_layer,
            heads=config.n_head,
            mlp_dim=config.mlp_dim,
            dim_head=config.dim_head,
        ).to('cpu')

        try:
            # Attempt to load the model
            print("Loading model...")
            encoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            encoder.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
        
        
        with h5py.File(self.file_path, 'r') as hf:
            lenth = hf['lenth'][()]
            for i in range(lenth):
                trajectory = hf[f'trajectory_{i}']
                obs_temp = trajectory['obs'][()] / 255.0
                actions = trajectory['actions'][()]
                r_e = trajectory['rewards'][()]
                r_c = trajectory['rewards_'][()]
                obs_temp = torch.from_numpy(obs_temp).float()
                states = encoder(obs_temp).detach().cpu().numpy()

                self.states.extend(states)
                self.actions.extend(actions)
                self.r_e.extend(r_e)
                self.r_c.extend(r_c)
                
        self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.r_e = np.array(self.r_e)
        self.r_c = np.array(self.r_c)
        
        del encoder
        if env_mode == 'left':
            self.actions = self.actions[:, 0:1]
        print("Data loading complete, number of trajectories:", lenth, "data size:", self.states.shape[0])

        
            
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        sample = {'states': self.states[idx], 'actions': self.actions[idx]}
        return sample
    
    def sample(self, batch_size):
        index = np.random.choice(range(self.__len__()), batch_size, replace=False)
        bn_s, bn_r_e, bn_r_c, bn_a = self.states[index], self.r_e[index], self.r_c[index], self.actions[index]
        return  bn_s, bn_r_e, bn_r_c, bn_a


