import socket
import sys
from pathlib import Path
from typing import Final
# from time import sleep
import select

import h5py
import cv2
import numpy as np

ROOT_PATH = Path(__file__).parents[1].absolute()
# package_path = '/home/wsl/pythonWork/paper_one/'
sys.path.append(str(ROOT_PATH / ''))
from config.config import load_config

AGENT_ID: Final[str] = "Agent"
WSL_PORT = 8066


def key_to_action(key: str):
    if key == 'up' or 'up' in key:
        return (1, 0)
    if key == 'down' or 'down' in key:
        return (-1, 0)
    if key == 'left' or 'left' in key:
        return (0, 1)
    if key == 'right' or 'right' in key:
        return (0, -1)
    return (0, 0)


def obs_to_image_array(obs: np.ndarray):
    image = obs[0:3, :, :].transpose(1, 2, 0).astype(np.uint8)
    # image = image.transpose(1, 2, 0)
    return image


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_mode', type=str, default='merge')  # merge or left_t
    parser.add_argument('--max_episode_steps', type=np.int64, default=400)
    parser.add_argument('--seed', type=np.int64, default=6)
    parser.add_argument('--reward_mode', type=str, default='intensive')
    parser.add_argument('--behavior_mode', type=str, default='radical')  # conservative or radical

    args = parser.parse_args()
    
    
    from smarts_env.env import make_env
    env = make_env(
        env_mode=args.env_mode,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        headless=False,
        reward_mode=args.reward_mode,
    )
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('172.26.5.84', WSL_PORT))
    server_socket.listen(1)

    print("等待连接...")
    # Accept a connection from a Windows client
    client_socket, address = server_socket.accept()
    print(f"Connection from: {address}")

    human_trajectory = []
    tr = {'obs': [], 'actions': [], 'rewards': [], 'next_obs': [], 'done': [], 'rewards_': [], 'speed': [], 'acceleration': []}
    num = 0
    # Receive and process keypress information
    try:
        obs, _ = env.reset()
        desired_speed = env.get_ego_now_speed()
        image = obs_to_image_array(obs)
        # Send this image to the client
        client_socket.send(image.tobytes())
        # Create an OpenCV window
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        r_sum = 0
        while True:
            # Display the current frame
            cv2.imshow('Video', image)
            cv2.waitKey(int(10))
            # Set the timeout to 10ms
            ready, _, _ = select.select([client_socket], [], [], 0.1)
            if client_socket in ready:
                data = client_socket.recv(1024)
                if not data:
                    break
                # Process keypress information here, perform corresponding actions as needed
                # print("Keypress info:", data.decode())
                key = data.decode()
            else:
                # If no data is received within 10ms, set key to an empty string
                key = ''
            
            # data = client_socket.recv(1024)
            # key = data.decode()
            # print("key:", key)
            # Process keypress information here, perform corresponding actions as needed
            act = key_to_action(key)
            desired_speed += act[0] * 4
            desired_speed = np.clip(desired_speed, 0, 20)
            act = ((desired_speed - 10) / 10, act[1])
            
            print("act:", act)
                
            obs_, reward, terminated, truncated, info = env.step(act)
            speed = env.get_ego_now_speed()
            acceleration = env.get_ego_now_acceleration()
            
            r_sum += reward

            tr['obs'].append(obs)
            tr['actions'].append(act)
            tr['rewards'].append(reward)
            tr['next_obs'].append(obs_)
            if terminated:
                tr['done'].append(1)
            else:
                tr['done'].append(0)
            tr['rewards_'].append(info['neighborhood_reward'])
            tr['speed'].append(speed)
            tr['acceleration'].append(acceleration)

            if terminated and info['reached_goal']:
                # Record trajectory upon reaching the goal
                human_trajectory.append(tr)
                obs, _ = env.reset()
                print("Completed, total reward:", r_sum)
                r_sum = 0
                tr = {'obs': [], 'actions': [], 'rewards': [], 'next_obs': [], 'done': [], 'rewards_': [], 'speed': [], 'acceleration': []}
                num += 1
                print("Completion number", num)
            elif terminated:
                obs, _ = env.reset()
                print("Failed, total reward:", r_sum)
                r_sum = 0
                tr = {'obs': [], 'actions': [], 'rewards': [], 'next_obs': [], 'done': [], 'rewards_': [], 'speed': [], 'acceleration': []}
            obs = obs_
            image = obs_to_image_array(obs)
    except (KeyboardInterrupt, ConnectionResetError):
        print("Program has exited")
    
    # save
    with h5py.File(f'data/{args.behavior_mode}_{args.env_mode}1_{num}.hdf5', 'w') as hf:
        hf.create_dataset('lenth', data=len(human_trajectory))
        for i, tr in enumerate(human_trajectory):
            group = hf.create_group(f'trajectory_{i}')
            group.create_dataset('obs', data=tr['obs'])
            group.create_dataset('actions', data=tr['actions'])
            group.create_dataset('rewards', data=tr['rewards'])
            group.create_dataset('next_obs', data=tr['next_obs'])
            group.create_dataset('done', data=tr['done'])
            group.create_dataset('rewards_', data=tr['rewards_'])
            group.create_dataset('speed', data=tr['speed'])
            group.create_dataset('acceleration', data=tr['acceleration'])
            
   
    cv2.destroyAllWindows()

    client_socket.close()
    server_socket.close()
    env.close()
