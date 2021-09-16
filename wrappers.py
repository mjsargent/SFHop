import gym 
import gym_minigrid
import numpy as np

class MTWrapper(gym.core.Wrapper):
    """
    Wrapper to allow for non multitask envs to 
    be run using the core loop defined in main
    """
    def __init__(self, env):
        super().__init__(env)
        self.tile_rewards = {"null": 0}
    
    def set_tile_rewards(self,*args):
        pass
    
    def possible_object_rewards(self):
        return 1

class MovePenaltyWrapper(gym.core.Wrapper):
    """
    Adds a small penalty to actions akin
    to some VizDoom mazes
    """

    def __init__(self, env, penalty = -0.01):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += self.penalty
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
