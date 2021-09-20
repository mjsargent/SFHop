import gym
import numpy as np
import torch
import random

from networks import *
from alg import *
from buffers import *
from wrappers import *

_COLOURS = ["blue", "red", "purple", "grey", "green", "yellow"]

def create_net(args, env: gym.Env):
    """
    create a net appropriate for a given rl algo, env and obs size
    """
    frames = 3
    #TODO add env type check to determine if frame stacking is needed (atari) 
    #TODO add support for passing remaining hyperparams
    if args.alg == "DQN":
        net = QNetwork(env, frames, args.lr)

    elif args.alg == "DQNReconstruct":
        net = QNetworkReconstruct(env, frames, args.lr)

    elif args.alg == "SF":
        net = SFNet(env, frames, args.lr, phi_dim = args.phi_dim,feature_loss=args.feature_loss)
    
    elif args.alg == "SFOnline":
        net = SFNetOnlineReward(env, frames, args.lr)

    elif args.alg == "SFMiniBatch":
        net = SFMiniBatch(env, frames, lr, phi_dim=args.phi_dim, feature_loss=args.feature_loss)
        
    elif args.alg == "SFHop":
        net = SFHopNet(env, frames, args.lr)


    else:
        pass
        #TODO add error handling

    return net


def create_env(env_name: str, task, fully_obs:bool = False, move_penalty = 0.0):
    _FOURROOMS_TASK_IDS = {"static":"MiniGrid-MTEnvFourRoomsStatic-v0", "shuffled":"MiniGrid-MTEnvFourRoomsShuffleLocations-v0", "wall_colour":"MiniGrid-MTEnvFourRoomsStaticWalls-v0", \
                       "landmarks":"MiniGrid-MTEnvFourRoomsLandmarks-v0"}

   
    if "MiniGrid" in env_name:
        import gym_minigrid
        from gym_minigrid.wrappers import ImgObsWrapper

        if "MiniGridFourRooms" in env_name:
            print(_FOURROOMS_TASK_IDS)
            gym_id = _FOURROOMS_TASK_IDS[task]
            env = gym.make(gym_id)
            # TODO if task variable is landmarks 
            #       set inital landmarks

        else:
            # accept a gym id directly if not using MT fourrooms
            gym_id = env_name

            env = gym.make(gym_id)
            env = MTWrapper(env)

        if fully_obs:
            from gym_minigrid.wrappers import FullyObsWrapper
            env = FullyObsWrapper(env)
        if move_penalty != 0:
            env = MovePenaltyWrapper(env, move_penalty)

        env = ImgObsWrapper(env)
        print(env)
        print(env.observation_space)
        return env

    else:
        pass
        #TODO error handling

def create_replay_buffer(buffer_limit, alg, gym_id, fully_obs, rollout_length = 16, p_reward = 0.3):
    if alg == "SFMiniBatch":
        return MiniBatchBuffer(buffer_limit, rollout_length)
    elif alg == "SF":
        return RewardPriorityBuffer(buffer_limit, p_reward)
    else:
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

def get_epsilon(start_e, end_e, duration, global_step, override_epsilon):
    if override_epsilon > 0:
        return override_epsilon
    else:
        return linear_schedule(start_e, end_e, duration, global_step)

def change_task(env, writer, task):
    #TODO check env type and choose task change appropriately
    if task == "static":
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

