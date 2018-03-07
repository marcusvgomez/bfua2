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
	def __init__(self, num_agents, num_actions, hidden_size = 15, 
				 activation_fn = nn.ReLU, K = 2, num_states = 5,
				 minibatch_size = 2, skip_connection = True):
		super(agent, self).__init__()
		assert activation_fn is not None
		self.num_agents = num_agents
		self.num_actions = num_actions
		self.K = K
		self.minibatch_size = minibatch_size
		self.hidden_size = hidden_size
		self.activation_fn = activation_fn()
		self.skip_connection = True


		self.stateEncoder = nn.Embedding(num_states, hidden_size)

		self.communication_cells = []
		self.hidden_cells = []

		for i in range(K):
			self.communication_cells.append(nn.Linear(hidden_size, hidden_size))
			self.hidden_cells.append(nn.Linear(hidden_size, hidden_size))

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
		state = inputs

		init_hidden = self.stateEncoder(state)
		temp_comm = init_hidden.sum(dim=1)
		curr_comm = temp_comm.view(self.minibatch_size, 1, self.hidden_size).repeat(1, self.num_agents, 1)
		curr_hidden = init_hidden

		for i in range(self.K):
			curr_comm = self.communication_cells[i](curr_comm)
			curr_hidden = self.hidden_cells[i](curr_hidden)

			if self.skip_connection:
				curr_hidden = self.activation_fn(curr_hidden + curr_comm + init_hidden)
			else:
				curr_hidden = self.activation_fn(curr_hidden + curr_comm)

			temp_comm = curr_hidden.sum(dim = 1)
			curr_comm = temp_comm.view(self.minibatch_size, 1, self.hidden_size).repeat(1, self.num_agents, 1)

		actions = self.hidden_to_actions(curr_hidden)
		advantage = self.hidden_to_advantage(curr_hidden)

		actions_softmax = self.softmax(actions) #probability of action for every agent

		actions_list = []
		for i in range(self.minibatch_size):
			m = Categorical(probs = actions_softmax[i])
			action = m.sample()
			log_prob = m.log_prob(action)
			actions_list.append((action, log_prob, actions_softmax[i]))


		return actions_list, advantage


def main():
	test = Variable(torch.Tensor([[1,2,3],[3,4,4]]))
	agent_trainable = agent(num_agents = num_agents, num_actions = num_actions)
	agent_trainable(test.long())

if __name__ == "__main__":
	main()