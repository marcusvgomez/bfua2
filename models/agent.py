'''
Agent implementation with no minibatching
'''

import sys
sys.path.append("../utils/")
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.distributions import Categorical

from env import STATE_DIM, ACTION_DIM
GOAL_DIM = 6
'''
Agent operating in the environment 


num_agents: number of agents in the environment
vocab_size: size of the vocabulary we're operating on
num_landmarks: number of landmarks in the environment
input_size: size of the vector representing the representation of other landmarks/agents
hidden_comm_size: hidden communication size for FC communication layer
comm_output_size: size representing the output communication vector
hidden_input_size: size of the hidden layer for the location data
input_output_size: size of the output from the FC layers for location data
hidden_output_size: hidden size of the output layer for the FC layer
memory_size: size of the memory bank
goal_size: size of the goal
is_cuda: are we using cuda
'''
GOAL_DIM = 6
class agent(nn.Module):
    def __init__(self, num_agents, vocab_size, num_landmarks,
                 input_size, hidden_comm_size, comm_output_size,
                 hidden_input_size, input_output_size,
                 hidden_output_size, action_dim = ACTION_DIM,
                 memory_size = 32, goal_size = 6, is_cuda = False, dropout_prob = 0.0,
                 is_goal_predicting = False, minibatch_size = 1024, num_gpus = 2):
        super(agent, self).__init__()
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.vocab_size = vocab_size
        self.memory_size = memory_size
        self.input_size = input_size
        self.dropout_prob = dropout_prob
        self.action_dim = action_dim
	self.use_cuda = is_cuda
        self.is_goal_predicting = is_goal_predicting
        self.comm_output_size = comm_output_size
        self.minibatch_size = minibatch_size
        self.num_gpus = num_gpus
        if self.is_goal_predicting: 
          self.goal_dim = GOAL_DIM 
        else:
          self.goal_dim = 0

        # dropout_prob = 0.0

        # print ("vocab size is: ", self.vocab_size + input_size)


        self.communication_FC = nn.Sequential(
                nn.Linear(vocab_size + memory_size, hidden_comm_size),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_comm_size, comm_output_size + self.goal_dim)
            )

        self.input_FC = nn.Sequential(
                nn.Linear(self.input_size * STATE_DIM, hidden_input_size),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_input_size, input_output_size)
            )

        self.output_FC = nn.Sequential(
                nn.Linear(input_output_size + comm_output_size + goal_size + memory_size, hidden_output_size),
                nn.ELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_output_size, action_dim + vocab_size + memory_size * num_agents + memory_size)
            )


        #activation functions and dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.tanh = nn.Tanh()
        self.gumbel_softmax = GumbelSoftmax(tau=1.0,use_cuda = is_cuda)

        self.embeddings = nn.Embedding(vocab_size, vocab_size)

        #if is_cuda:
        #    self.initializeCuda()

    '''
    Takes in inputs which is a tuple containing
    X: (N + M) x input_size matrix that represents the coordinates of other agents/landmarks
    C: N x communication_size matrix that represents the communication of all other agents
    g: goal vector always represented in R^3
    M: N x memory_size communcation memory matrix N by memory_size by N 
    m: state memory 
    Runs a forward pass of the neural network spitting out the action and communication actions
    '''
    def forward(self, inputs):
        
        X, C, g, M, m, is_training = inputs
        #reshaping everything for sanity
        M = M.transpose(2, 3)
        C = C.transpose(1, 2)
        X = X.transpose(1, 2)
        g = g.transpose(1, 2)
        m = m.transpose(1, 2)

        C = C.repeat(self.num_agents, 1, 1, 1)
        X = X.unsqueeze(1)  
        X = X.repeat(1, self.num_agents, 1, 1)

        C = C.transpose(0, 1)

        # print M, C, X, g, m

        communication_input = torch.cat([C, M], 3) #concatenate along the first direction

        #(comm_size + goal_size) x ___ x N
        #(comm_size) x __ x N, (goal) x __ x N

        if not self.is_goal_predicting: 
          comm_out = self.communication_FC(communication_input)
          comm_pool = self.softmaxPool(comm_out)
          goal_out = None
        else:
          comm_out = self.communication_FC(communication_input)
          comm_results = []
          goal_results = []
          for i in range(self.num_agents):
            comm_i = comm_out[:,i,:,0:self.comm_output_size]
            goal_i = comm_out[:,i,:,self.comm_output_size:]
            comm_results.append(torch.unsqueeze(comm_i, 0))
            goal_results.append(torch.unsqueeze(goal_i, 0))
          comm_intermediate = torch.cat(comm_results, 0)
          comm_pool = self.softmaxPool(comm_intermediate)
          goal_out = torch.cat(goal_results, 0)
          goal_out = goal_out.transpose(2,3)
          goal_out = goal_out.transpose(0,1)

        

        # print X
        loc_output = self.input_FC(X)
        # print X, loc_output
        loc_pool = self.softmaxPool(loc_output, dim = 2).squeeze() #this is bad for now need to fix later
        # print loc_pool
        # assert False


        #concatenation of pooled communication, location, goal, and memory
        output_input = torch.cat([comm_pool, m, loc_pool, g], 2)

        output = self.output_FC(output_input)

        psi_u, psi_c, mem_mm_delta, mem_delta = output[:, :, :self.action_dim], output[:, :, self.action_dim:self.action_dim + self.vocab_size],\
                                                output[:, :, self.action_dim + self.vocab_size: self.action_dim + self.vocab_size + self.memory_size * self.num_agents],\
                                                output[:, :, self.action_dim + self.vocab_size + self.memory_size * self.num_agents: ]

        if is_training:
            epsilon_noise = make_epsilon_noise()
            if self.use_cuda:
                epsilon_noise = epsilon_noise.cuda()
            action_output = psi_u + epsilon_noise
        else:
            action_output = psi_u
        # print action_output.min(), action_output.max()

#        mem_mm_delta = mem_mm_delta.view(self.num_agents, self.memory_size, -1)#self.num_agents)
        # mem_mm_delta = mem_mm_delta.contiguous().view(self.minibatch_size, -1, self.memory_size, self.num_agents)
        mem_mm_delta = mem_mm_delta.contiguous().view(self.memory_size, self.num_agents, self.num_agents, -1)
        mem_mm_delta = mem_mm_delta.transpose(3, 2)
        mem_mm_delta = mem_mm_delta.transpose(2, 1)
        mem_mm_delta = mem_mm_delta.transpose(0, 1)
        mem_mm_delta = mem_mm_delta.transpose(1,2)
        mem_mm_delta = mem_mm_delta.transpose(2,3)
        # mem_delta = mem_mm_delta.transpose()
        # mem_mm_delta = mem_mm_delta.contiguous().view(self.minibatch_size, self.memory_size, self.num_agents, -1)
        # mem_mm_delta = mem_mm_delta.transpose(3,2)
        # mem_mm_delta = mem_mm_delta.transpose(2,1)
        # mem_mm_delta = mem_mm_delta.transpose(2,3)


        temp_comm_output = self.gumbel_softmax(psi_c)
        if is_training:
            communication_output = self.gumbel_softmax(psi_c)
        else:
            psi_c_log = self.softmax(psi_c)
            cat = Categorical(probs=psi_c_log)
            comm_one_hot = cat.sample()
            communication_output = torch.zeros(self.num_agents, self.vocab_size)
            for i, val in enumerate(comm_one_hot.data):
                communication_output[i][val] = 1.
            communication_output = Variable(communication_output).cuda()


        #memory updates
        if is_training:
            M_eps = make_epsilon_noise()
            m_eps = make_epsilon_noise()
            if self.use_cuda:
                M_eps = M_eps.cuda()
                m_eps = m_eps.cuda()
            # print M, mem_mm_delta, M_eps
            # print self.tanh(M + mem_mm_delta + M_eps)
            M = self.tanh(M + mem_mm_delta + M_eps).transpose(2, 3)
            m = self.tanh(m + mem_delta + M_eps).transpose(1, 2)
        
        else:
            M = self.tanh(M.transpose(2,3) + mem_mm_delta)
            m = self.tanh(m + mem_delta).transpose(1,2)

        # M = self.tanh(M.transpose(1,2) + mem_mm_delta)
        # m = self.tanh(m + mem_delta).transpose(0,1)

        #transposing because we have to i think
        #I really need to check to make sure math stuff works
        action_output = action_output.transpose(1,2)
        communication_output = communication_output.transpose(1,2)
       
        #print action_output, communication_output, M, m, goal_out
        return action_output, communication_output, M, m, goal_out

    '''
    Runs a softmax pool which is taking the softmax for all entries
    then returning the mean probabilities 
    '''
    def softmaxPool(self, inputs, dim = 0):
        input_prob = self.softmax(inputs)
        return torch.mean(input_prob, dim = dim)


    # I need to check this code
    def initializeCuda(self):
        # print "initializing Cuda"
        for param in self.parameter:
            print (param)



