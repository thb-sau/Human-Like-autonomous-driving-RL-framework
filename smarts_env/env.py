from typing import Any, Dict
from pathlib import Path

import gymnasium as gym

from smarts.sstudio.scenario_construction import build_scenarios
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent_interface import NeighborhoodVehicles, RGB
from smarts.sstudio.scenario_construction import build_scenarios


ROOT_PATH = Path(__file__).parents[0].absolute()
# sys.path.append(ROOT_PATH)

def make_env(env_mode='left', max_episode_steps=600, seed=2024, headless=False, reward_mode='intensive'):
    # print("=============================================")
    from smarts_env.env_process import Preprocess
    from stable_baselines3.common.monitor import Monitor

    assert env_mode in ['left', 'left_t', 'lane_change', 'merge'], "env_mode must be 'left' or 'left_t' or 'lane_change' or 'merge'"

    agent_interfaces = {
        "Agent": AgentInterface.from_type(
            AgentType.LanerWithSpeed,  
            max_episode_steps=max_episode_steps,
            waypoint_paths=True,
            top_down_rgb=RGB(),
            neighborhood_vehicle_states=NeighborhoodVehicles(radius=60),
        )
    }
    if env_mode == 'left':
        agent_type = 'speed'
        scenario = "scenarios/sumo/intersections/1_to_1lane_left_turn_c_agents_1"

    elif env_mode == 'left_t':
        agent_type = 'laner_speed'
        scenario = "scenarios/sumo/intersections/1_to_2lane_left_turn_t_agents_1"

    elif env_mode == 'lane_change':
        agent_type = 'laner_speed'
        scenario = "scenarios/sumo/straight/3lane_cruise_agents_1"

    elif env_mode == 'merge':
        agent_type = 'laner_speed'
        scenario = "scenarios/sumo/merge/3lane_agents_1" 
    
    # 配置场景、代理接口和运行模式
    scenarios = [str(ROOT_PATH / Path(scenario))]
    build_scenarios(scenarios=scenarios)
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        seed=seed,
        headless=headless,  # If False, enables Envision display.
        # scenarios_order=ScenarioOrder.sequential,
    )
    env = SingleAgent(env)
    env = Preprocess(env, agent_interface=list(agent_interfaces.values())[0], reward_mode=reward_mode, agent_type=agent_type)
    env = Monitor(env)
    return env

