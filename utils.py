import gym
import numpy as np
import torch
import random

from networks import *
from buffers import *

_COLOURS = ["blue", "red", "purple", "grey", "green", "yellow"]

def create_net(alg: str, env: gym.Env, lr: float):
    """
    create a net appropriate for a given rl algo, env and obs size
    """
    frames = 3
    #TODO add env type check to determine if frame stacking is needed (atari) 
    if alg == "DQN":
        net = QNetwork(env, frames, lr)

    elif alg == "SF":
        net = SFNet(env, frames, lr)
    
    elif alg == "SFOnline":
        net = SFNetOnlineReward(env, frames, lr)
        
    elif alg == "SF_Hop":
        net = SFHopNet(env, frames, lr)


    else:
        pass
        #TODO add error handling

    return net

def create_env(env_name: str, task, fully_obs:bool = False):
    
    if "MiniGrid" in env_name:
        import gym_minigrid
        from gym_minigrid.wrappers import ImgObsWrapper
        env = gym.make(env_name)
        # TODO if task variable is landmarks 
        #       set inital landmarks

        if fully_obs:
            from gym_minigrid.wrappers import FullyObsWrapper
            env = FullyObsWrapper(env)

        env = ImgObsWrapper(env)
        return env
    else:
        pass
        #TODO error handling

def create_replay_buffer(buffer_limit, alg, gym_id, fully_obs):
    return ReplayBuffer(buffer_limit)
     
def set_seeds(env,seed, torch_deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
#
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def change_task(env, writer, task):
    #TODO check env type and choose task change appropriately
    if task == "default":
        env.set_tile_rewards()
        writer.add_scalars("charts/tile_rewards", env.tile_rewards)

    elif task == "wall_colour":
        env.set_tile_rewards()
        env.set_wall_colour(random.choice(_COLOURS))
        writer.add_scalars("charts/tile_rewards", env.tile_rewards)
        #TODO record wall colour?

    elif task == "landmarks":
        env.set_tile_rewards()
        # TODO write method for moving landmarks
        writer.add_scalars("charts/tile_rewards", env.tile_rewards)
    return env
