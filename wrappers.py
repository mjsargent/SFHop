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
