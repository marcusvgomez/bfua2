'''
Harness testing the agents for now since I can't tell if it's differentiable
'''


import sys
sys.path.append("./utils/")
sys.path.append("./models/")
from utils import *
from agent import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

num_agents = 1
num_landmarks = 3
vocab_size = 20
input_size = 57
hidden_comm_size = 17
comm_output_size = 32
hidden_input_size = 45
input_output_size = 64
hidden_output_size = 84


def initializeWeights(model):
	for name, param in model.named_parameters(): 
		if "embeddings" in name:
			# print (name)
			param.data.copy_(torch.eye(vocab_size))

	# for name, param in model.named_parameters():
		# if "embeddings" in name:
			# print (param)

def agentDifferentiable():
	# X = Variable(torch.Tensor(torch.randn((num_agents, input_size, hidden_input_size))))
	X = Variable(torch.Tensor(torch.randn((1, num_agents + num_landmarks, input_size))))
	C = Variable(torch.Tensor(torch.randn(num_agents, vocab_size)))
	g = Variable(torch.Tensor(torch.randn(3)))
	M = Variable(torch.Tensor(torch.randn(num_agents, 32)))
	m = Variable(torch.Tensor(torch.randn(32)))


	currAgent = agent(num_agents, vocab_size,
				num_landmarks, input_size, hidden_comm_size, comm_output_size,
				hidden_input_size, input_output_size,
				hidden_output_size,
				memory_size = 32, goal_size = 3, is_cuda = False, dropout_prob = 0.1)

	initializeWeights(currAgent)

	optimizer = optim.Adam(currAgent.parameters(), lr = 0.05)

	_, embedding = currAgent((X, C, g, M, m, True))
	print (embedding)
	m = embedding.shape
	#m, n = embedding.shape
	#print (m, n)

	# loss = (((embedding.float() - Variable(torch.Tensor(torch.randn(1)))))**2).sum()
	loss = ((embedding.float() - Variable(torch.Tensor(torch.randn((m)))))**2).sum()
	#print (loss)

	# assert(False)


	for name, param in currAgent.named_parameters():
		print (name, param)
		break

	loss.backward()
	optimizer.step()

	for name, param in currAgent.named_parameters():
		print (name, param)
		break




	# next_embedding = currAgent()


def main():
	agentDifferentiable()

if __name__ == "__main__":
	main()
