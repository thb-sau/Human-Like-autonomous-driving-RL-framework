import gymnasium as gym
import numpy as np
from PIL import Image
import math

from smarts_env.filter_obs import FilterObs

class Preprocess(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_interface: gym.Space, reward_mode: str = 'sparse', agent_type="speed"):
        super().__init__(env)
        # 断言reward_mode="intensive" 或者 "sparse"
        assert reward_mode in ['intensive', 'sparse', 'default'], "reward_mode='intensive' or 'sparse'"
        self.reward_mode = reward_mode

        assert agent_type in ['speed', 'laner_speed'], "agent_type='speed' or 'laner_speed'"
        self.agent_type = agent_type
        
        self.filter_obs = FilterObs(
            top_down_rgb=agent_interface.top_down_rgb,
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(9, 80, 80),
            dtype=np.uint8,
        )
        if agent_type == 'laner_speed':
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1]),
                high=np.array([1, 1]),
                shape=(2,),
                dtype=np.float32,
            )
        elif agent_type == 'speed':
            self.action_space = gym.spaces.Box(
                low=np.array([-1,]),
                high=np.array([1,]),
                shape=(1,),
                dtype=np.float32,
            )

        self.obs_seq = [None, None, None]

        # initialize desired speed and lane
        self.desired_speed = 2
        self.lane_change = 0
        self.lane_index = 0

        # 上一步的原始观测值
        self.last_obs = None

    def add_obs(self, obs):
        assert len(self.obs_seq) == 3 and self.obs_seq[0] is not None and self.obs_seq[1] is not None and self.obs_seq[2] is not None
        self.obs_seq[0] = self.obs_seq[1]
        self.obs_seq[1] = self.obs_seq[2]
        self.obs_seq[2] = obs

    def get_obs(self):
        finall_obs = np.zeros((9, 80, 80))
        # 将三张图片依次叠加在一起
        for i, image in enumerate(self.obs_seq):
            finall_obs[i*3:i*3+3, :, :] = image
        return finall_obs
    
    def act(self, obs, action):
        a_1 = -1 if action[0] <= -0.8 else action[0]
        if len(action) > 1:
            a_2 = 1 if action[1] > 0.5 else -1 if action[1] < -0.5 else 0
        else:
            a_2 = 0
        
        self.desired_speed = (a_1 + 1) * 10
        
        ego_state = obs["ego_vehicle_state"]
        wp_paths = obs["waypoint_paths"]
        ego_lane_index = ego_state["lane_index"]

        # self.lane_index = ego_lane_index + self.lane_change
        # if a_2 != 0:
        #     self.lane_change = a_2
        #     self.lane_index += self.lane_change
        self.lane_change = a_2 
        self.lane_index += self.lane_change

        self.lane_index = np.clip(self.lane_index, 0, len(wp_paths)-1)

        if self.lane_index == ego_lane_index:
            self.lane_change = 0

        self.desired_speed = np.clip(self.desired_speed, 0, 20)
        speed = ego_state['speed']
        if self.agent_type == 'laner_speed':
            return (np.float32(self.desired_speed - speed), np.int8(self.lane_change))
        elif self.agent_type == 'speed':
            return (np.float32(self.desired_speed - speed), np.int8(ego_lane_index))
        

    def step(self, action):
        assert self.last_obs is not None
        action = self.act(self.last_obs, action)
        # print(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_mode == 'intensive':
            obs, reward, terminated, truncated, info = self._intensive_reward(obs, reward, terminated, truncated, info)
        elif self.reward_mode == 'sparse':
            obs, reward, terminated, truncated, info = self._sparse_reward(obs, reward, terminated, truncated, info)
        self.last_obs = obs
        obs = self.filter_obs.filter(obs)
        self.add_obs(obs=obs)
        finall_obs = self.get_obs()
        
        return finall_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # 可以在reset时进行一些特定的处理
        obs, info = self.env.reset(**kwargs)

        ego_state = obs["ego_vehicle_state"]
        
        speed = ego_state['speed']
        self.desired_speed = speed
        self.ego_now_speed = speed

        ego_lane_index = ego_state["lane_index"]
        self.lane_index = ego_lane_index

        self.last_obs = obs
        obs = self.filter_obs.filter(obs)
        self.obs_seq = [obs, obs, obs]
        finall_obs = self.get_obs()
        return finall_obs, info
    
    def _sparse_reward(self, obs, reward, terminated, truncated, info):
        info['reached_goal'] = False
        # 重置奖励
        reward = np.float64(0)
        # 获取自我车辆信息
        ego_vehicle_info = obs["ego_vehicle_state"]
        ego_position = ego_vehicle_info["position"][0:2]
        neighborhood_vehicle_states = obs["neighborhood_vehicle_states"]
        positions = neighborhood_vehicle_states["position"][:, 0:2]
        non_zero_rows = np.any(positions != 0., axis=1)
        positions = positions[non_zero_rows]
        distances = np.sqrt(np.sum((positions - ego_position)**2, axis=1))

        self.linear_acceleration = ego_vehicle_info['linear_acceleration']
        self.yaw_rate = ego_vehicle_info['yaw_rate']

        # 获取测含量速度
        ego_speed = ego_vehicle_info['speed']
        self.ego_now_speed = ego_speed
        
        neighborhood_speed = neighborhood_vehicle_states['speed'][non_zero_rows]
        neighborhood_reward = np.sum(np.float64(neighborhood_speed * 0.001) / np.exp(1e-4 * distances))
        info['neighborhood_reward'] = neighborhood_reward
        # 如果撞击
        if obs["events"]["collisions"]:
            reward += np.float64(-1)
            terminated = True
            print(f"Collided.")

        # 如果车辆偏离道路
        if obs["events"]["off_road"]:
            terminated = True
            print(f"Went off road.")

        if obs["events"]["reached_goal"]:
            reward += np.float64(1)
            info['reached_goal'] = True
            print(f"reached_goal.")

        if obs["events"]["not_moving"]:
            terminated = True
            print(f"not_moving.")
        return obs, reward, terminated, truncated, info
    
    def _intensive_reward(self, obs, reward, terminated, truncated, info):
        info['reached_goal'] = False
        # 重置奖励
        reward = np.float64(0)
        # 获取自我车辆信息
        ego_vehicle_info = obs["ego_vehicle_state"]
        ego_position = ego_vehicle_info["position"][0:2]
        neighborhood_vehicle_states = obs["neighborhood_vehicle_states"]
        positions = neighborhood_vehicle_states["position"][:, 0:2]
        non_zero_rows = np.any(positions != 0., axis=1)
        positions = positions[non_zero_rows]
        distances = np.sqrt(np.sum((positions - ego_position)**2, axis=1))

        self.linear_acceleration = ego_vehicle_info['linear_acceleration']
        self.yaw_rate = ego_vehicle_info['yaw_rate']

        # 获取测含量速度
        ego_speed = ego_vehicle_info['speed']
        self.ego_now_speed = ego_speed
        ego_speed = np.clip(ego_speed, 0, 20)
        reward += np.float64(ego_speed * 0.001)

        neighborhood_speed = neighborhood_vehicle_states['speed'][non_zero_rows]
        neighborhood_reward = np.sum(np.float64(neighborhood_speed * 0.001) / np.exp(1e-4 * distances))
        info['neighborhood_reward'] = neighborhood_reward
        
        # 如果撞击
        if obs["events"]["collisions"]:
            reward += np.float64(-1)
            terminated = True
            print(f"Collided.")

        # 如果车辆偏离道路
        if obs["events"]["off_road"]:
            reward += np.float64(-1)
            terminated = True
            print(f"Went off road.")

        if obs["events"]["reached_goal"]:
            reward += np.float64(1)
            info['reached_goal'] = True
            print(f"reached_goal.")

        if obs["events"]["not_moving"]:
            terminated = True
            print(f"not_moving.")
        
        # print(f"reward: {reward}")
        return obs, reward, terminated, truncated, info


    def get_ego_now_speed(self):
        """
        m/s
        """
        return self.ego_now_speed
    
    def get_ego_now_acceleration(self):
        """
        m/s^2
        """
        x, y, _ = self.linear_acceleration
        return math.sqrt(x**2 + y**2)
    
    def get_ego_now_yaw_rate(self):
        """
        rad/s
        """
        return self.yaw_rate