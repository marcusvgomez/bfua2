import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from agent import *
from env import Levers, Traffic

from agent import agent


class Controller:
    def __init__(self, runtime_config):#, runtime_config):
        #self.M = runtime_config.num_agents
        #self.A = runtime_config.num_actions
        #self.minibatch_size = runtime_config.minibatch_size

        self.M = 500
        self.A = 5
        self.minibatch_size = 64
        self.agent = agent
        self.use_cuda = True
        self.is_supervised = False
        self.sparse_communication = True
        self.num_comm_channels = 1

        self.agent_trainable = self.agent(num_agents = self.M, num_actions = self.A, 
                                          minibatch_size = self.minibatch_size, use_cuda=self.use_cuda, 
                                          sparse_communication = self.sparse_communication)
        self.agent_trainable#.cuda()
        if self.use_cuda: self.agent_trainable.cuda()

        self.initializeParameters()

        if runtime_config.env == "traffic":
            self.env = Traffic()
        else:
            self.env = Levers(num_agents=self.M, num_samples=self.A, minibatch_size = self.minibatch_size) 


    def initializeParameters(self):
        return
            #for m in self.agent_trainable():
                #if isinstance(m, nn.Embedding):
                #    m.weight.

    #runs the agents for a given set of states
    #the states should be of shape (minibatch_size, num_agents)
    #MAKE SURE THE SHAPING IS RIGHT 
    def run(self):
        def make_supervised(states):
            new_states = []
            for i in range(self.minibatch_size):
                curr_states = states[i].numpy()
                sorted_states = curr_states.copy()
                sorted_states.sort()
                curr_new_states = []
                for state in curr_states:
                    idx = np.where(sorted_states==state)
                    curr_new_states.append(idx[0][0])

                new_states.append(curr_new_states)
            return torch.Tensor(new_states)

        states = self.env.get_initial_state()
        if self.is_supervised:
            states = make_supervised(states)
        if self.sparse_communication:
            active_comm_channels = self.make_sparse_pairing(states, num_channels = self.num_comm_channels)

        states = Variable(states)

        if self.use_cuda: states = states.cuda()
            
        if self.sparse_communication:
            action_list, advantage = self.agent_trainable((states.long(), active_comm_channels))
        else:
            action_list, advantage = self.agent_trainable(states.long())
        reward = self.env.get_reward(action_list)
        
        return self.compute_loss(reward, action_list, advantage), reward.mean()

    def make_sparse_pairing(self, states, num_channels):
        assert (num_channels < self.A)
        agent_pairing = []
        _ , agent_space = states.shape
        for i in range(self.minibatch_size):
            curr_pairing = {}
            curr_state = states.numpy()[i].tolist()
            curr_state_set = set(curr_state)
            for j in range(agent_space):
                state_val = curr_state[j]
                curr_state_set.remove(state_val)
                curr_pairing[state_val] = np.random.choice(list(curr_state_set), size = num_channels, replace = False)
                curr_state_set.add(state_val)
            agent_pairing.append(curr_pairing)
        return agent_pairing


    def compute_loss(self, reward, action_list, advantage, alpha = 0.03):
        loss = 0.
        print (reward.mean())
        for i in range(self.minibatch_size):
            currReward = torch.Tensor([reward[i]])
            currAdvantage = advantage[i]

            cRewardCuda = currReward#.cuda()
            if self.use_cuda: cRewardCuda = cRewardCuda.cuda()
            prob_scaling = Variable(cRewardCuda - currAdvantage.data, requires_grad = False)#.cuda()
            if self.use_cuda: prob_scaling = prob_scaling.cuda()
            prob_loss = (action_list[i][1] * prob_scaling).sum()

#                print action_list[i][1]
            
            currReward_var = Variable(currReward, requires_grad = True)#.cuda()
            if self.use_cuda: currReward_var = currReward_var.cuda()
            reward_loss = alpha * (currReward_var - currAdvantage)**2
            
            loss += (prob_loss - reward_loss)
        return -loss/self.minibatch_size



class TrafficController:
    def __init__(self, runtime_config):#, runtime_config):
        #self.M = runtime_config.num_agents
        #self.A = runtime_config.num_actions
        #self.minibatch_size = runtime_config.minibatch_size

        self.game_length = 40
        self.maxN = 10
        self.A = 2
        self.minibatch_size = 288
        self.agent = agent
        self.use_cuda = True
        self.thread_minibatches = True #make this a flag later

        if self.thread_minibatches:
            self.num_threads = 8
            self.minibatch_size //= self.num_threads

        self.agent_trainable = self.agent(num_agents = self.maxN, num_actions = self.A, minibatch_size = self.minibatch_size, is_traffic=True, use_cuda=self.use_cuda)
        if self.use_cuda: self.agent_trainable.cuda()

        self.initializeParameters()

        #self.env = Levers(num_agents=self.M, num_samples=self.A, minibatch_size = self.minibatch_size) 
        self.env = Traffic(max_agents=self.maxN, p_next=0.05, minibatch_size = self.minibatch_size)

    def initializeParameters(self):
        return
            #for m in self.agent_trainable():
                #if isinstance(m, nn.Embedding):
                #    m.weight.

    #runs the agents for a given set of states
    #the states should be of shape (minibatch_size, num_agents)
    #MAKE SURE THE SHAPING IS RIGHT 
    def run(self):
        state = Variable(torch.Tensor(self.env.step_init()))
        if self.use_cuda: state = state.cuda()
        rewards = []
        actions = []
        advantages = []
        for iter_ in range(self.game_length):
            #print "ITERATION: ", iter_
            action_list, advantage = self.agent_trainable(state.float())
            actions_only_l = [elem[0] for elem in action_list]
            actions_np = np.zeros((self.minibatch_size, self.maxN))
            for i in range(len(actions_only_l)):
                for j in range(len(actions_only_l[i])):
                    actions_np[i,j] = actions_only_l[i][j]
            reward, next_state = self.env.step_forward(actions_np)
            rewards.append(reward)
            actions.append(action_list)
            advantages.append(advantage)
            
        return self.compute_loss(rewards, actions, advantages)
    def run_thread_minibatches(self): 
        state = Variable(torch.Tensor(self.env.step_init()))
        if self.use_cuda: state = state.cuda()
        rewards = []
        actions = []
        advantages = []
        for iter_ in range(self.game_length):
            #print "ITERATION: ", iter_
            action_list, advantage = self.agent_trainable(state.float())
            actions_only_l = [elem[0] for elem in action_list]
            actions_np = np.zeros((self.minibatch_size, self.maxN))
            for i in range(len(actions_only_l)):
                for j in range(len(actions_only_l[i])):
                    actions_np[i,j] = actions_only_l[i][j]
            reward, next_state = self.env.step_forward(actions_np)
            rewards.append(reward)
            actions.append(action_list)
            advantages.append(advantage)
            
        return self.compute_loss(rewards, actions, advantages)



    def compute_loss(self, rewards, actions, advantages):
        b = 0.0
        for iter_ in range(self.game_length):
            b += self.compute_loss_at_t(rewards[iter_], actions[iter_], advantages[iter_])
        return b
    def update_at_epoch(self, epoch):
        self.env.update_p_next(epoch)

    def compute_loss_at_t(self, reward, action_list, advantage, alpha = 0.03):
        loss = 0.
        #print reward.mean()
        for i in range(self.minibatch_size):
            currReward = torch.Tensor([reward[i]])
            currAdvantage = advantage[i]

            if self.use_cuda: currReward = currReward.cuda()
            prob_scaling = Variable(currReward - currAdvantage.data, requires_grad = False)#.cuda()
            if self.use_cuda: prob_scaling = prob_scaling.cuda() 

            prob_loss = (action_list[i][1] * prob_scaling).sum()

            
            currReward_var = Variable(currReward, requires_grad = True)#.cuda()
            if self.use_cuda: currReward_var.cuda()
            reward_loss = alpha * (currReward_var - currAdvantage)**2
            
            loss += (prob_loss - reward_loss)
        return -loss/self.minibatch_size
def main():
    runtime_config = None
    curr_controller = Controller(runtime_config)
    curr_controller.run()
    curr_controller = TrafficController(runtime_config)
    curr_controller.run()
    test = np.array([[1,2,3],[3,4,4]])


if __name__ == "__main__":
    main()
