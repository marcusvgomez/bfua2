import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from agent import *



from agent import agent


class Controller:
	def __init__(self):#, runtime_config):
		# self.M = runtime_config.num_agents
		# self.A = runtime_config.num_actions
		# self.minibatch_size = runtime_config.minibatch_size

		self.M = 3
		self.A = 6
		self.minibatch_size = 2

		self.agent = agent

		self.initialize_agent()


	#initializes a trainable agent
	def initialize_agent(self):
		self.agent_trainable = self.agent(num_agents = self.M, num_actions = self.A, minibatch_size = self.minibatch_size)


	#runs the agents for a given set of states
	#the states should be of shape (minibatch_size, num_agents)
	#MAKE SURE THE SHAPING IS RIGHT 
	def runAgents(self, states):
		states = Variable(torch.Tensor(states))
		actionList, advantage = self.agent_trainable(states.long())
		print actionList

def main():
	curr_controller = Controller()
	test = np.array([[1,2,3],[3,4,4]])
	curr_controller.runAgents(test)


if __name__ == "__main__":
	main()