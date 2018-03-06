import numpy as np
import torch

import torch.nn.functional as F
from torch.autograd import Variable

'''
Takes log probabilities and outputs and outputs the gummel softmax for sampling
Taken from:
https://github.com/pytorch/pytorch/issues/639
def gumbel_softmax(prob, tau = 1):
    noise = Variable(make_gummel_noise(prob))
    x = (prob + noise) / tau
    x = F.softmax(x.view(prob.size(0), -1))
    return x.view_as(prob)

Makes gummel noise as described in the paper

'''
class GumbelSoftmax(torch.nn.Module):
    def __init__(self, tau = 1.0, use_cuda = True):
        super(GumbelSoftmax, self).__init__()
        self.use_cuda = use_cuda
        self.softmax = torch.nn.Softmax()
        self.tau = tau

    def forward(self, x):
        if self.use_cuda:
            U = Variable(torch.rand(x.size()).cuda(), requires_grad = True)
        else:
            U = Variable(torch.rand(x.size()), requires_grad = True)
        out = x - torch.log(-torch.log(U + 1e-10) + 1e-10)
        ret = self.softmax(out / self.tau)
        return ret 

def make_epsilon_noise():
    return -torch.log(-torch.log(Variable(torch.Tensor([(1,)]).uniform_(0,1))))