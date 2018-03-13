import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd
import random

import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.distributions import Categorical
import numpy as np
from multiprocessing import Pool
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

def step_forward_worker(elem):
    idx, mb_actions, mb_agents, old_state = elem
    colls = np.zeros((14,14))
    reward = 0.0
    new_agents = []
    agents = mb_agents[:]
    for agent in agents:
        idx, loc, route, t, remaining_steps = agent
        r_steps = remaining_steps[:]
        t = t+1
        action = mb_actions[idx]
        new_loc = loc
        if action == 1:
            next_step = r_steps[0]
            r_steps.pop(0)
            new_loc = [None, None]
            new_loc[0] = loc[0] + next_step[0]
            new_loc[1] = loc[1] + next_step[1]
            new_loc = tuple(new_loc)
            colls[new_loc[0], new_loc[1]] += 1.0
            old_state[loc[0], loc[1]] -= 1
            old_state[new_loc[0], new_loc[1]] += 1
        reward += (-0.01*t)
        if len(r_steps) > 0: new_agents.append((idx, new_loc, route, t, r_steps))
    C_t = 0
    for i in range(14):
        for j in range(14):
            if colls[i,j] > 1: C_t += 1
    reward += (-10.0)*C_t
    return idx, reward, new_agents, old_state
class Traffic:
    def __init__(self, max_agents=10, p_next=0.05, minibatch_size=2):
        self.Nmax = max_agents
        self.p_next = p_next
        self.minibatch_size = minibatch_size
        self.entries = [(13,7), (0,6), (7,0), (6,13)]
        self.generate_cache()
        self.reset()
        self.pool = Pool(8)

    def reset(self):
        self.state = np.zeros((self.minibatch_size, 14,14))
        self.N = [0]*self.minibatch_size
        self.agents = [[] for _ in range(self.minibatch_size)]
#        self.unused_ids = [range(self.Nmax)]*self.minibatch_size
        self.unused_ids = [[i for i in range(self.Nmax)] for _ in range(self.minibatch_size)]

    def update_p_next(self, epoch_num):
        if epoch_num < 100: return
        elif epoch_num > 200: self.p_next = 0.2
        else:
            self.p_next = (0.15)*(epoch_num - 100)/float(100) + 0.05
        
    def gen_at_point(self, point, mb):
        unused_ids = self.unused_ids[mb][:]
        if self.N[mb] < self.Nmax:
            p = random.random()
            if p <= self.p_next:
                self.N[mb] = self.N[mb] + 1
                self.state[mb, point[0], point[1]] += 1.0
                idx = random.choice(unused_ids)
                unused_ids.remove(idx)
                route = random.randint(0,2)
                cur = self.agents[mb][:]
                cur.append((idx, point, route, 0, self.cache[(point, route)]))
                self.agents[mb] = cur
        self.unused_ids[mb] = unused_ids
        return

    ##doing MB, n, 211 
    def format_states(self, mb):
        to_ret = np.zeros((self.Nmax, 218))
        for agent in self.agents[mb]:
            idx, loc, route, _, _ = agent
            r = np.zeros((3))
            r[route] = 1.0
            l = np.zeros((196))
            l[loc[0]*14 + loc[1]] = 1.0
            n = np.zeros((self.Nmax))
            n[idx] = 1.0
            visible = np.zeros((3,3))
            for i in range(-1, 2, 1):
                if (i + loc[0]) < 0: continue 
                if (i + loc[0]) > 13: continue
                for j in range(-1,2,1):
                  if (j + loc[1]) < 0: continue
                  if(j + loc[1]) > 13: continue
                  visible[(i+1),(j+1)] = self.state[mb, i + loc[0], j+loc[1]]
            visible_flat = visible.flatten()
            state_a = np.concatenate([visible_flat,n,l,r])
            to_ret[idx,:] = state_a
        return to_ret 
    ##returns list of MB lists of states of size 3^2 * |n| * |l| * |r| = 9 * 10 * 196(??) * 3
    def step_init(self):
        ##four entry points are (7,13), (6,0), (0,6), and (13, 7)
        states = np.zeros((self.minibatch_size, self.Nmax, 218))
        for mb in range(self.minibatch_size):
          for entry in self.entries: 
              self.gen_at_point(entry, mb)
          mb_state = self.format_states(mb)
          states[mb,:,:] = mb_state
        return states

    def generate_cache(self):
        cache = {}
        cache[((13,7),0)] = [(-1,0) for _ in range(7)] + [(0,-1) for _ in range(7)]
        cache[((13,7),1)] = [(-1,0) for _ in range(13)]
        cache[((13,7),2)] = [(-1,0) for _ in range(6)] + [(0,1) for _ in range(6)]
        
        cache[((0,6),0)] = [(1,0) for _ in range(7)] + [(0,1) for _ in range(7)]
        cache[((0,6),1)] = [(1,0) for _ in range(13)]
        cache[((0,6),2)] = [(1,0) for _ in range(6)] + [(0,-1) for _ in range(6)]

        cache[((7,0),0)] = [(0,1) for _ in range(7)] + [(-1,0) for _ in range(7)]
        cache[((7,0),1)] = [(0,1) for _ in range(13)]
        cache[((7,0),2)] = [(0,1) for _ in range(6)] + [(1,0) for _ in range(6)]

        cache[((6,13),0)] = [(0,-1) for _ in range(7)] + [(1,0) for _ in range(6)]
        cache[((6,13),1)] = [(0,-1) for _ in range(13)]
        cache[((6,13),2)] = [(0,-1) for _ in range(6)] + [(-1,0) for _ in range(6)]
        self.cache = cache

    ##takes in actions, returns reward
    ##actions should be MB x max_agents

    def step_forward_multiprocess(self, actions):
        rewards = np.zeros((self.minibatch_size))
        new_states = np.zeros((self.minibatch_size, self.Nmax, 218))
        idx = xrange(self.minibatch_size)
        mb_actions = [actions[mb,:] for mb in idx]
        mb_agents = [self.agents[mb][:] for mb in idx]
        old_states = [self.state[mb,:,:] for mb in idx]
        vals = []
        for mb in idx: vals.append((mb, mb_actions[mb], mb_agents[mb], old_states[mb]))
        self.pool.map(step_forward_worker, vals)

    def step_forward(self, actions):
        return self.step_forward_multiprocess(actions)
        ### L O FUCKING L
        rewards = np.zeros((self.minibatch_size))
        new_states = np.zeros((self.minibatch_size, self.Nmax, 218))
        for mb in range(self.minibatch_size):
          colls = np.zeros((14,14))
          reward = 0.0
          new_agents = []
          mb_agents = self.agents[mb][:]
          for agent in mb_agents:
              idx, loc, route, t, remaining_steps = agent
              r_steps = remaining_steps[:]
              t = t+1
              action = actions[mb, idx]
              new_loc = loc
              if action == 1:
                  next_step = r_steps[0]
                  r_steps.pop(0)
                  new_loc = [None, None]
                  new_loc[0] = loc[0] + next_step[0]
                  new_loc[1] = loc[1] + next_step[1]
                  new_loc = tuple(new_loc)
                  colls[new_loc[0], new_loc[1]] += 1.0
                  self.state[mb, loc[0], loc[1]] -= 1
                  self.state[mb, new_loc[0], new_loc[1]] += 1
              reward += (-0.01*t)
              if len(r_steps) > 0: new_agents.append((idx, new_loc, route, t, r_steps))
          C_t = 0
          for i in range(14):
              for j in range(14):
                  if colls[i,j] > 1: C_t += 1
          reward += (-10.0)*C_t
          rewards[mb] = reward
          self.agents[mb] = new_agents
          new_state_mb = self.format_states(mb)
          new_states[mb,:,:] = new_state_mb
        return rewards, new_states


def main():
    traffic_sim = Traffic(minibatch_size=288)
    actions = np.ones((288,10))
    print (traffic_sim.step_init()[0][0].shape)
    for i in range(40): traffic_sim.step_forward(actions)

if __name__ == "__main__": main()
