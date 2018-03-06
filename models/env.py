'''
Environment implementation

Takes as input the actions of each agent at a given time step
and outputs the observation of the state of the world from 
the perspective of each agent. Internally maintains the current
state of the world.
'''
import torch
from torch.autograd import Function, Variable

R_DIM = 2
STATE_DIM = 6# 2(for position) + 2(for velocity) + 2(for gaze) + NO(1(for color))
ACTION_DIM = 4 # 2 (for velocity) + 2(for gaze)

class env:
    def __init__(self, num_agents, num_landmarks, timestep=0.1, damping_coef=0.5, is_cuda = False, minibatch_size=1024, num_gpus = 2):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.timestep = timestep #delta t
        self.damping_coef = damping_coef # this is 1 - gamma
        self.gamma = 1 - self.damping_coef
        self.minibatch_size = minibatch_size
        self.num_gpus = 2
        self.is_cuda = is_cuda
        self.initialize_state()


    def initialize_state(self):
        self.world_state_agents = torch.FloatTensor(self.minibatch_size, STATE_DIM, self.num_agents).zero_()
        ##agent 0 is at (0,0)
        self.world_state_agents[:,0,1] = 10.0 ##init agent 1 at (10,10) 
        self.world_state_agents[:,1,1] = 10.0
        self.world_state_agents[:,0,2] = -10.0
        self.world_state_agents[:,1,2] = -10.0
        self.world_state_landmarks = torch.FloatTensor(self.minibatch_size, STATE_DIM, self.num_landmarks).zero_()

        
        ## this probably shouldn't be hardcoded but...whatever
        self.transform_L = torch.FloatTensor(STATE_DIM, STATE_DIM).zero_()
        self.transform_L[0:2,0:2] = torch.eye(R_DIM)
        self.transform_L[0:2,2:4] = self.timestep * torch.eye(R_DIM)
        self.transform_L[2:4,2:4] = self.gamma * torch.eye(R_DIM)
        # self.transform_L[6,6] = 1.
        self.transform_R = torch.FloatTensor(STATE_DIM, ACTION_DIM).zero_()
        self.transform_R[2:4,0:2] = self.timestep * torch.eye(R_DIM)
        self.transform_R[4:6,2:4] = torch.eye(R_DIM)

        self.transform_L = Variable(self.transform_L, requires_grad = True)
        self.transform_R = Variable(self.transform_R, requires_grad = True)
        self.world_state_agents = Variable(self.world_state_agents, requires_grad = True)
        self.world_state_landmarks = Variable(self.world_state_landmarks, requires_grad = True)


        if self.is_cuda:
            self.transform_L = self.transform_L.cuda()
            self.transform_R = self.transform_R.cuda()
            self.world_state_agents = self.world_state_agents.cuda()
            self.world_state_landmarks = self.world_state_landmarks.cuda()

    def modify_world_state(self, agents, landmarks):
        pass

    def clear(self):
        del self.transform_L
        del self.transform_R
        del self.world_state_agents
        del self.world_state_landmarks
        self.initialize_state()

    ##this is literally horrible design
    def expose_world_state(self): return self.world_state_agents, self.world_state_landmarks
    
    def expose_world_state_extended(self):
        # returns a version of the agents world state that is duplicated to be (STATE_DIM * NUM_AGENTS) by NUM_AGENTS to serve as the observations of all of the agents
        self.total_state = torch.FloatTensor(self.minibatch_size, STATE_DIM * self.num_agents, self.num_agents).zero_()
        # bad but not sure if pytorch has convenient reshaping tools to do this
        for c in range(self.num_agents):
            agent_state = self.world_state_agents[:, c].data
            for r in range(self.num_agents):
                self.total_state[:, STATE_DIM*r:STATE_DIM*(r+1), c] = self.world_state_agents[:, :, r].data
        return self.total_state
        
    
    ##actions should be a 6 X N tensor
    ##returns a 12(N+M) x N tensor
    def forward(self, actions):
        L = torch.matmul(self.transform_L, self.world_state_agents)
        R = torch.matmul(self.transform_R, actions)
        self.world_state_agents = L + R
        #self.world_state_agents[0:2,:] = torch.clamp(self.world_state_agents[0:2,:], -10.0, 10.0)
        result = torch.FloatTensor(self.minibatch_size, STATE_DIM*(self.num_agents + self.num_landmarks), self.num_agents)
        for i in range(self.num_agents):
            row = torch.FloatTensor(self.minibatch_size, STATE_DIM*(self.num_agents + self.num_landmarks))
            for j in range(self.num_agents):
                # print self.world_state_agents.data[:,:,j]
                # print self.world_state_agents.data
                row[:,STATE_DIM*j:STATE_DIM*(j+1)] = self.world_state_agents.data[:,:,j] 
            offset = STATE_DIM*self.num_agents
            for j in range(self.num_landmarks):
                row[:,offset + STATE_DIM*j: offset + STATE_DIM*(j+1)] = self.world_state_landmarks.data[:,:,j]

            """
            for j in range(self.num_agents):
                row[STATE_DIM*j:STATE_DIM*(j+1)] = self.world_state_agents[:,j] - self.world_state_agents[:,i]
                row[STATE_DIM*j + 6] = self.world_state_agents[6,j]
            offset = STATE_DIM*self.num_agents
            for j in range(self.num_landmarks):
                row[offset + STATE_DIM*j: offset + STATE_DIM*(j+1)] = self.world_state_landmarks[:,j] - self.world_state_agents[:,i]
		
		##gaze is always technically 0, so unsure if that means i should make it constant
		##this is probably the right thing to do, but leaving it uncommented for now
		##we can use their visualization script to see what's happening lol
		
                #row[offset + STATE_DIM*j + 6] = 0.0
                #row[offset + STATE_DIM*j + 7] = 0.0
                #row[offset + STATE_DIM*j + 8] = 0.0
		

		##color is constant and not relative
                row[offset + STATE_DIM*j + 6] = self.world_state_landmarks[6,j]
            """
            result[:,:,i] = row
        return Variable(result, requires_grad = True)
        
