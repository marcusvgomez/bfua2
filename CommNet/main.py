# from utils.utils import *
from agent import * 
from controller import *
from env import *
from config import *
#torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function, Variable
import torch.autograd as autograd

import torch.nn.init as init
from torch.nn.parameter import Parameter

#other imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import shutil
from itertools import *
import gc

def getTime():
    return datetime.datetime.now().strftime("%d-%H-%M-%s")

#save_path = '/cvgl2/u/bcui/cs234/results/'
save_path = "./results/"
# model_name = 'communication_vision_non_adversarial'
model_name = 'levers_communication'
loss_name = save_path + model_name + "loss.npy"
reward_name = save_path + model_name + "accuracy.npy"

currTime = getTime()
print (currTime)
save_model_name = save_path + model_name + " date " + currTime + ".pt"
best_name = save_path + "best/" + model_name + " date " + currTime + ".pt"
loss_dir = save_path + "loss/" + model_name + " date " + currTime + ".pt"


#model saving code
def save_model(model, optimizer, epoch_num, best_dev_acc, modelName = save_model_name, bestModelName = best_name, is_best = False):
    state = {'epoch': epoch_num + 1,
             'state_dict': model.state_dict(),
             'best_dev_acc': best_dev_acc,
             'optimizer': optimizer.state_dict()
            }
    torch.save(state, modelName)
    if is_best:
        shutil.copyfile(modelName, bestModelName)


#updates the optimizer if we are going to do decay rates
def updateOptimizer(optimizer, decay_rate = 5):
    print ("optimizing parameters")
    for param_group in optimizer.param_groups:
        param_group['lr'] /= decay_rate
        print (param_group['lr'])
    return optimizer

#plotting loss
def plot_loss(loss, ylabel = 'Loss', typing = 'Loss_'):
    x_axis = [i for i in range(len(loss))]
    y_axis = loss
    plt.plot(x_axis, y_axis, label = 'o')
    plt.xlabel('Epoch Number')
    plt.ylabel(ylabel)
    plt.savefig(typing + str(model_name) + '.png')

def load_model(modelPath, controller, optimizer):
    print "Loading Checkpoint ... =>"
    checkpoint = torch.load(modelPath)
    controller.agent_trainable.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print "best dev acc is: ", checkpoint['best_dev_acc']

def main():
    parser = argparse.ArgumentParser(description="Train time babbbyyyyyyyy")
    parser.add_argument('--n-epochs', '-e', type=int, help="Optional param specifying the number of epochs")
    parser.add_argument('--use_cuda', action='store_true', help="Optional param that specifies whether to train on cuda")
    parser.add_argument('--test', action='store_true', help="Optional param that specifies whether to train on cuda")
    parser.add_argument('--load-model', type=str, help='Optional param that specifies model weights to start using')
    parser.add_argument('--save-model', type=str, help='Optional param that specifies where to save model weights')
    parser.add_argument('--save-model-epoch', type=int, help='Optional param that specifies where to save model weights')

    parser.add_argument('--num-agents', type=int, help='Optimal param that specifies number of agents')
    parser.add_argument('--hidden-size', type=int, help='Optional param that specifies number of hidden')
    parser.add_argument('--comm-steps', type=int, help='Optional param that specifies number of comm steps')
    parser.add_argument('--minibatch-size', type=int, help='Optional param that specifies the minibatch size')
    parser.add_argument('--num-gpus', type=int, help='Optional param that specifies the number of GPUs')


    arg_dict = vars(parser.parse_args())

    runtime_config = RuntimeConfig(arg_dict)
    print runtime_config.num_agents

    #this needs to be fixed
    controller = Controller(runtime_config)
    parameters = ifilter(lambda p :p.requires_grad, controller.agent_trainable.parameters())
    optimizer = optim.Adam(parameters, lr = 0.0005)
#    optimizer = optim.Adam(controller.agent_trainable.parameters())
    
    ## TODO: write train code

#    for j, param in enumerate(controller.agent_trainable.parameters()):
#        print param, j
#    assert False
    loss_list = []
    reward_list = []
    for i in range(58):

        loss, reward = controller.run()
        print loss
        loss_list.append(loss.data[0])
        reward_list.append(reward.data[0])

        if i % 100 == 0:
            print "EPOCH IS: ", i, " and loss is: ", loss.data[0], "reward is: ", reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print reward_list
    print loss_list

    np.save(loss_name, np.array(loss_list))
    np.save(reward_name, np.array(reward_list))
    plot_loss(loss_list)
    plot_loss(reward_list, ylabel = 'reward', typing = 'Reward_')
    

if __name__ == "__main__":
    main()

