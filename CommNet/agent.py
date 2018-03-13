import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.distributions import Categorical

num_agents = 3
num_actions = 4


'''
Agent module in pytorch for the commNet implemented in "Learning Multiagent Communication with Backpropagation"

Here the input variables are:
num_agents: number of agents in the environment
num_actions: number of actions in the environment
hidden_size: size of the hidden state we are producing (I assumed it's identical across all embedding/hidden spaces)
activation_fn: the activation function used to recompute next hidden state
K: number of communication timesteps
num_states: number of states the agent can start in, used for the embedding space
minibatch_size: number of minibatches we're running simultaneously
skip_connection: uses skip connections
'''
class agent(nn.Module):
    def __init__(self, num_agents, num_actions, hidden_size = 9,
                 activation_fn = nn.ReLU, K = 2, num_states = 5,
                 minibatch_size = 2, skip_connection = True, use_cuda = True, is_traffic=False,
                 is_supervised = True, sparse_communication = True):
                super(agent, self).__init__()
                assert activation_fn is not None
                self.num_states = num_states
                self.num_agents = num_agents
                self.num_actions = num_actions
                self.K = K
                self.minibatch_size = minibatch_size
                self.hidden_size = hidden_size
                self.activation_fn = activation_fn()
                self.skip_connection = True
                self.use_cuda = use_cuda
                self.is_traffic = is_traffic
                self.is_supervised = is_supervised
                self.sparse_communication = sparse_communication


                self.stateEncoder = nn.Embedding(num_agents, num_agents)
                self.stateEncoder.weight.data = torch.eye(num_agents, num_agents)
                self.stateEncoder.weight.requires_grad = False

                if is_traffic:
                    self.stateEncoder = nn.Linear(218, hidden_size)
                self.communication_cells = []
                self.hidden_cells = []
                self.skip_matrices = []
                
                for i in range(K+1):
                    if i == 0:
                        if not is_traffic:
                            self.communication_cells.append(nn.Linear(num_agents, hidden_size))
                            self.hidden_cells.append(nn.Linear(num_agents, hidden_size))
                        else:
                            self.communication_cells.append(nn.Linear(hidden_size, hidden_size))
                            self.hidden_cells.append(nn.Linear(hidden_size, hidden_size))
                    else:
                        print ("I is: ", i)
                        self.communication_cells.append(nn.Linear(hidden_size, hidden_size))
                        self.hidden_cells.append(nn.Linear(hidden_size, hidden_size))
                        self.skip_matrices.append(nn.Sequential(
                                nn.Linear(3*hidden_size, 2*hidden_size),
                                nn.ReLU(),
                                nn.Linear(2*hidden_size, hidden_size),
                                nn.ReLU()
                                ))

                self.communication_cells = nn.ModuleList(self.communication_cells)
                self.hidden_cells = nn.ModuleList(self.hidden_cells)
                self.skip_matrices = nn.ModuleList(self.skip_matrices)
                self.hidden_to_actions = nn.Linear(hidden_size, num_actions)
                self.hidden_to_advantage = nn.Linear(hidden_size, 1) #produce a scalar for all
                self.softmax = nn.Softmax(dim = 2)



    '''
    returns a list of actions which is of length minibatch_size
    in every entry in the list it has:
    a list of actions for every agent, the log_prob of taking the action
    and

    '''
    def forward(self, inputs):
        if self.sparse_communication:
            state, self.sparse_map = inputs
        else:
            state = inputs
        init_hidden = self.stateEncoder(state)
        init_hidden = self.hidden_cells[0](init_hidden)
        temp_comm = init_hidden.sum(dim = 1)
#                curr_comm = temp_comm.view(self.minibatch_size, 1, self.hidden_size).repeat(1, self.num_actions, 1)
#                curr_comm -= init_hidden
#                curr_comm /= (self.num_actions-1)
        if self.is_traffic:
            curr_comm = Variable(torch.zeros(self.minibatch_size, self.num_agents, self.hidden_size))
            if self.use_cuda: curr_comm = curr_comm.cuda()
        else:
            curr_comm = Variable(torch.zeros(self.minibatch_size, self.num_actions, self.hidden_size))#.cuda()
            if self.use_cuda: curr_comm = curr_comm.cuda()


        curr_hidden = init_hidden


        for i in range(1, self.K+1):
            curr_comm = self.communication_cells[i](curr_comm)
            curr_hidden = self.hidden_cells[i](curr_hidden)

            if self.skip_connection:
                #curr_hidden = self.activation_fn(self.skip_matrices[i-1](self.activation_fn(curr_hidden + curr_comm + init_hidden)))
                curr_hidden = self.activation_fn(self.skip_matrices[i-1](torch.cat([curr_hidden, curr_comm, init_hidden], dim = 2)))

            else: pass

            if self.sparse_communication:
                self.update_sparse_communication(curr_hidden)
            else:
                temp_comm = curr_hidden.sum(dim = 1)
                if self.is_traffic:
                    curr_comm = temp_comm.view(self.minibatch_size, 1, self.hidden_size).repeat(1, self.num_agents, 1)
                else:
                    curr_comm = temp_comm.view(self.minibatch_size, 1, self.hidden_size).repeat(1, self.num_actions, 1)
                    curr_comm -= curr_hidden
                    curr_comm /= (self.num_actions-1)
                    # for levers game
                    #if not self.is_traffic:
                    #    curr_comm = Variable(torch.zeros(self.minibatch_size, self.num_actions, self.hidden_size))#.cuda()
                    #    if self.use_cuda: curr_comm = curr_comm.cuda()

        actions = self.hidden_to_actions(curr_hidden)
        advantage = self.hidden_to_advantage(curr_hidden)
        advantage = advantage.mean(dim = 1)

        actions_softmax = self.softmax(actions) #probability of action for every agent

        actions_list = []
        for i in range(self.minibatch_size):
            m = Categorical(probs = actions_softmax[i])
            action = m.sample()
            log_prob = m.log_prob(action)
            actions_list.append((action.data.cpu().numpy(), log_prob, actions_softmax[i]))
                        #print action.data.cpu().numpy()


        return actions_list, advantage

    def update_sparse_communication(self, curr_hidden):
        update_comm = Variable(torch.zeros((self.minibatch_size, self.num_actions, self.num_actions)), requires_grad = True)
        if self.use_cuda: 
            update_comm = update_comm.cuda()
        for i in range(self.minibatch_size):
            curr_mapping = self.sparse_map[i]
            minibatch_agent_lever = curr_mapping.keys()
            for idx, agent_index in enumerate(curr_mapping):
                try:
                    for channel_index in curr_mapping[agent_index]:
                        channel_index = int(channel_index)
                        channel_index = minibatch_agent_lever.index(channel_index)
                        update_comm[i, idx, channel_index] = 1 
                except KeyError:
                    print "currmapping is: ", curr_mapping
                    print "agent index is: ", agent_index
                    assert False
        temp_comm = torch.bmm(update_comm, curr_hidden)


        return temp_comm


def main():
    test = Variable(torch.Tensor([[1,2,3],[3,4,4]]))
    agent_trainable = agent(num_agents = num_agents, num_actions = num_actions)
    agent_trainable(test.long())

if __name__ == "__main__":
    main()
