import collections
import random
import numpy as np

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)


class ReplayBufferSF(ReplayBuffer):
    #TODO determine if this is even needed
    def __init__(self, buffer_limit):
        super(ReplayBufferSF, self).__init__(buffer_limit)

    def sample(self, n):

        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

class MiniBatchBuffer():
    """
    buffer for storing short rollouts
    """
    def __init__(self, buffer_limit, rollout_length):
        # minibatches are determined by rollout_length not buffer_limit
        self.buffer_limit = buffer_limit
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.rollout_length = rollout_length
        self.ready = False

    def put(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) == self.rollout_length:
            self.ready = True

    def sample(self, n):
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in self.buffer:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

    def flush(self):
        self.ready = False
        self.buffer = collections.deque(maxlen=self.buffer_limit)

class RewardPriorityBuffer():
    """
    keeps rewarded transitions and non-rewarded transisions in
    seperate buffers, and samples from them according to 
    some p
    """
    def __init__(self,buffer_limit, p_reward):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.buffer_r = collections.deque(maxlen=buffer_limit)
        self.p_reward = p_reward

    def put(self, transition):
        # != instead of > to keep negative rewards
        if transition[2] != 0:
            self.buffer_r.append(transition)
        else:
            self.buffer.append(transition)

    def sample(self, n):
        # number of rewarded trans
        rewarded_sample = sum(np.random.uniform(0,1,n) < self.p_reward)
        #make sure this isn't larger than the number of trans in the rewarded buffer

        rewarded_sample = len(self.buffer_r) if rewarded_sample > len(self.buffer_r)  else rewarded_sample
        non_rewarded_sample = n - rewarded_sample 

        # check for the case where an bonus/penalty has been added
        if len(self.buffer) == 0:
            mini_batch = None
        else:
            mini_batch = random.sample(self.buffer, non_rewarded_sample)


        mini_batch_r = random.sample(self.buffer_r, rewarded_sample)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        if mini_batch is not None:
            for transition in mini_batch:
                s, a, r, s_prime, done_mask = transition
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_mask_lst.append(done_mask)

        for transition in mini_batch_r:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)


class NullBuffer():
    """
    null buffer that has the required methods but does not store anything
    """
    def __init_(self, *args):
        self.buffer = None

    def put(self, transition):
        pass

    # you really shouldn't be sampling from this but regardless
    def sample(self, n):
        return None

