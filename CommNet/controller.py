import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
from agent import *
from env import Levers
from env import Traffic 

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

		self.agent_trainable = self.agent(num_agents = self.M, num_actions = self.A, minibatch_size = self.minibatch_size)
                self.agent_trainable.cuda()

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
            states = Variable(self.env.get_initial_state()).cuda()
            action_list, advantage = self.agent_trainable(states.long())
            reward = self.env.get_reward(action_list)
            
            return self.compute_loss(reward, action_list, advantage)

        def compute_loss(self, reward, action_list, advantage, alpha = 0.03):
            loss = 0.
            print reward.mean()
#            for i in range(self.minibatch_size):
#                currReward = Variable(torch.Tensor([reward[i]]), requires_grad = False)
#                loss += (-action_list[i][1] * currReward).sum()
            for i in range(self.minibatch_size):
                currReward = torch.Tensor([reward[i]])
                currAdvantage = advantage[i]


                prob_scaling = Variable(currReward.cuda() - currAdvantage.data, requires_grad = False).cuda()
                prob_loss = (action_list[i][1] * prob_scaling).sum()

#                print action_list[i][1]
                
                currReward_var = Variable(currReward, requires_grad = True).cuda()
                reward_loss = alpha * (currReward_var - currAdvantage)**2
                
                loss += (prob_loss - reward_loss)
            return -loss/self.minibatch_size


def main():
        runtime_config = None
	curr_controller = Controller(runtime_config)
	test = np.array([[1,2,3],[3,4,4]])
	curr_controller.run()


if __name__ == "__main__":
	main()
