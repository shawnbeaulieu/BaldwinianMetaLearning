from __future__ import division

import os
import copy
import math
import glob
import torch
import random
import pickle
import multiprocessing

import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD, Adam
from torch.nn.modules.loss import CrossEntropyLoss

from task import OmniglotTask, MNISTTask
from dataset import Omniglot, MNIST
from inner_loop import InnerLoop
from omniglot_net import OmniglotNet
from score import *
from data_loading import *

import numpy as np
import pandas as pd

def load_pickle(filename):
    with open(filename, 'rb') as f:
        struct = pickle.load(f)
    return(struct)

def Get_Task(root, n_cl, n_inst, split='train'):
    if 'mnist' in root:
        return MNISTTask(root, n_cl, n_inst, split)
    elif 'omniglot' in root:
        return OmniglotTask(root, n_cl, n_inst, split)
    else:
       print('Unknown dataset')
    raise(Exception)


def Evaluate(individual):

    """
    Adapted inner loop of Model Agnostic Meta-Learning to be Baldwinian
    a la Fernando et al. 2018.
    Source: https://github.com/katerakelly/pytorch-maml/blob/master/src/maml.py
    
    """

    #tasks['t1'] = self.get_task("../data/{}".format(self.dataset), self.num_classes, self.num_inst)

    inner_net = InnerLoop(5, CrossEntropyLoss(), 3, 0.01, 100, 10, 3)

    for t in range(10):

        task = Get_Task("../data/{}".format('omniglot'), 5, 10)

        # Outer-loop is completed by NES for G generations

        inner_net.copy_weights(individual['network'])
        metrics = inner_net.forward(task)

        # Want validation accuracy for fitness (tr_loss, tr_acc, val_loss, val_acc):  
        print(metrics)

def Main():

    individual = load_pickle(glob.glob("BML_Repo_Ranked_ShortQ/Champions/1/Champ1_G53_*")[0])
    Evaluate(individual)


if __name__ == '__main__':
    Main()
    
