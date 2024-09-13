import h5py
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import  write_array_to_txt


file_path = '/home/wsl/pythonWork/paper_one/data/merge1_20.hdf5'
writer = SummaryWriter('./expert_merge_condition/')

def tensorboard_writer(data, name):
    # 将一维数组写入TensorBoard
    for i, value in enumerate(data):
        writer.add_scalar(name, value, i)
        
# 从HDF5文件中读取数据
with h5py.File(file_path, 'r') as hf:
    lenth = hf['lenth'][()]
    for i in range(lenth):
        trajectory = hf[f'trajectory_{i}']
        speed = trajectory['speed'][()]
        acceleration = trajectory['acceleration'][()]
        
        # speed = np.ndarray(speed)
        # acceleration = np.ndarray(acceleration)
        tensorboard_writer(speed, f'ep{i}/speed')
        tensorboard_writer(acceleration, f'ep{i}/acceleration')

        


        