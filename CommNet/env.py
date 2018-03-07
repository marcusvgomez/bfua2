import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.distributions import Categorical
import numpy as np
class Levers:
    def __init__(self, num_agents=10, num_samples=3, minibatch_size=2):
        self.N = num_agents
        self.m = num_samples
        self.minibatch_size = minibatch_size

    ## returns thing of size minibatch, m
    def get_initial_state(self):
        state = np.zeros((self.minibatch_size, self.m))
        for i in range(self.minibatch_size): 
            state[i,:] = np.random.choice(self.N, self.m)
        state = torch.Tensor(state)
        return state


    ##actions should be minibatch, m  where action[i,:] is list of levers pulled
    ##returns rewards of size minibatch
    def get_reward(self, actions):
        reward = np.zeros(self.minibatch_size,)
        for i in range(self.minibatch_size):
            reward[i] = 1.*len(np.unique(actions[i][0])) / self.m
        return reward
