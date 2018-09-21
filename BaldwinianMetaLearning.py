#!/usr/bin/env python3.6
# BaldwinianMetaLearning.py
# Author: Shawn Beaulieu
# September 11th, 2018

"""
Skeletal implementation of "Meta-Learning by the Baldwin Effect" by Fernando et al.

"""

from __future__ import division

import os
import copy
import math
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

from functools import partial
#from pathos.multiprocessing import ProcessingPool as Pool

#First population of parents

def pickle_dict(dictionary, filename):
    p = pickle.Pickler(open("{0}.p".format(filename),"wb"))
    p.fast = True
    p.dump(dictionary)

def Preserve(data, filename):
    #df_data.to_csv(filename, mode='a', sep=",", header=None, index=None)
    df_data = pd.DataFrame(data).T
    try:
        df_data.to_csv(filename, mode='a', sep=",", header=None, index=None)
    except:
        # If no such file exists (new experiment) create it:
        df_data.to_csv(filename, sep=",", header=None, index=None)

def Generate(mean, std, dimensions, lower, upper, dist='gauss'):
    if dist == 'gauss':
        return(np.clip(np.random.normal(mean, std, dimensions), lower, upper))
    elif dist == 'exp':
        return(np.clip(np.random.exponential(mean, dimensions), lower, upper))

def Mean_Gradient(z, u, s=1.0):
    return((z-u)/s**2)

def Sigma_Gradient(z, u, s):
    return(((z-u)**2 - s**2)/s**3)

def Compute_Gradients(pop_of_weights, scores):
    """
    Population based metric for approximating the gradient with respect to
    network generating parameters (NES). Layer-wise computation of 'gradients'
    """

    # First compute mean and standard deviation across network features:
    # Per parameter gradient depends on mean of that parameter across the
    # population

    means = np.mean(np.array(pop_of_weights), axis=0)
    #stds = np.std(np.array(pop_of_weights), axis=0)

    # Initialize approximate gradients
    means_grad = np.zeros_like(pop_of_weights[0])
    #stds_grad = np.zeros_like(pop_of_weights[0])

    for idx in range(len(pop_of_weights)):

        # Gradient is the mean of gradients across the population, where each 
        # individual gradient is weighted by the ranked score:

        means_grad = np.add(means_grad, Mean_Gradient(pop_of_weights[idx], means)*scores[idx], 
                                out=means_grad, casting='unsafe')
        #stds_gradient += Sigma_Gradient(individual[0], means, stds)*individual[1]

    # Turn sum in mean:
    means_grad = np.divide(means_grad, len(pop_of_weights), out=means_grad, casting='unsafe')
    #stds_gradient /= len(pop_of_weights)

    return(means, means_grad)

def Rank(population):

    scores = [p['fitness'] for p in population]

    ranks = np.argsort(scores)
    champ_idx = ranks[-1]
    adjusted_scores = np.linspace(-0.5, 0.5, len(population))

    for idx in range(len(population)):

        rank = ranks[idx]
        population[rank]['ranked_fitness'] = adjusted_scores[idx]
        print(population[rank]['fitness'], adjusted_scores[idx])

    return(population, champ_idx)

def Sample_W(m,v):
    return(np.random.normal(m,v))

def Inspect_Swarm(population, ranked=True):

    mean_grads = {}
    means = {}

    if ranked:
        scores = np.linspace(-0.5, 0.5, len(population))

    else:
        scores = np.array([p['fitness'] for p in population])

    layers = population[0]['network'].state_dict().keys()

    def convert2numpy(weights, layer):
        return(weights.state_dict()[layer].data.numpy())

    for layer in layers:

        # Take weights in network and convert them to a numpy array for computing gradients
        weights_in_layer = [convert2numpy(p['network'], layer) for p in population]

        # Compute mean, std, and corresponding gradients:
        means[layer], mean_grads[layer] = Compute_Gradients(weights_in_layer, scores)

        #a,b,c,d = Compute_Gradients(weights_in_layer, scores)
        #(means[layer], stds[layer], mean_grads[layer], std_grads[layer]) = (a,b,c,d)

    return(means, mean_grads)


def mkdir(pathway):

        try:
            os.makedirs(pathway)
        except OSError:
            if not os.path.isdir(pathway):
                raise

class NES():

    """
    Natural evolution strategies for Baldwinian meta-learning
    """
    P = {}

    def __init__(self, parameters={}):

        # Use dictionary of parameters to class variables of the same name:
        self.__dict__.update(NES.P, **parameters)

        self.directory = os.getcwd()
        self.data_repo = "{0}/{1}".format(self.directory, self.folder)
        mkdir(self.data_repo)

        # Set up multiprocessing:     
        #cpus = multiprocessing.cpu_count()
        #self.pool = Pool(cpus-1)

        # Spawn population of networks using self.blueprint as a template:
        self.loss_fn = CrossEntropyLoss()
        self.population = self.Initialize_Population()

        # Begin!
        self.Evolve()

    def Evolve(self):

        self.g = 0

        for _ in range(self.generations):

            #self.pool.map(self.Evaluate, self.population)

            """==================== EVALUATE ====================""" 

            for individual in self.population:
                self.Evaluate(individual)

            if self.ranked_fitness:
                # Rank scores (batch normalization):
                self.population, champ_idx = Rank(self.population)

            else:
 
                champ_idx = np.argsort([p['fitness'] for p in self.population])[-1]

            """==================== NEW CHAMPION ====================""" 

            if self.g == 0:
                self.champion = copy.copy(self.population[champ_idx])

            else:
                if self.champion['fitness'] < self.population[champ_idx]['fitness']:
                    self.champion = copy.copy(self.population[champ_idx])

            print("Generation {0}: High Score = {1}".format(self.g, self.champion['fitness']))

            self.Save_Champion()

            """==================== NES GRADIENT ===================="""

            # Compute approximate gradients:
            self.means, self.mean_grads = Inspect_Swarm(self.population)

            """==================== UPDATE POPULATION ====================""" 

            # Replenish swarm  
            self.New_Swarm()
            self.g += 1


        print("End of evolution: High Score = {0}".format(self.champion['fitness']))


    def Initialize_Population(self):

        """
        Initialize master nets containing parameters governing the distributions
        from which individual weights are drawn in the OmniglotNet.

        """

        return_list = []

        for p in range(self.popsize):

            new_master_net = {}

            num_input_channels = 1 if self.dataset == 'mnist' else 3
            net = OmniglotNet(self.num_classes, self.loss_fn, num_input_channels)

            new_master_net['network'] = net
            new_master_net['fitness'] = 0.0 


            keys = ("_tr_acc", "_tr_loss", "_val_loss", "_val_acc")

            for task in range(self.meta_batch_size):

                for k in keys:
                    
                    key = "t" + str(task) + k
                    new_master_net[key] = 0.0

            return_list.append(new_master_net)

        return(return_list)

    def Evaluate(self, individual):

        """
        Adapted inner loop of Model Agnostic Meta-Learning to be Baldwinian
        a la Fernando et al. 2018.
        Source: https://github.com/katerakelly/pytorch-maml/blob/master/src/maml.py
    
        """

        inner_net = InnerLoop(self.num_classes, self.loss_fn, self.num_updates, self.inner_step_size, 
                                self.inner_batch_size, self.meta_batch_size, self.num_input_channels)

        #tasks = {}
        #tasks['t1'] = self.get_task("../data/{}".format(self.dataset), self.num_classes, self.num_inst)
        
        for it in range(self.meta_batch_size):
 
            task = self.get_task("../data/{}".format(self.dataset), self.num_classes, self.num_inst)

            # Outer-loop is completed by NES for G generations

            inner_net.copy_weights(individual['network'])
            metrics = inner_net.forward(task)
            
            # Want validation accuracy for fitness (tr_loss, tr_acc, val_loss, val_acc):  
            individual['fitness'] += metrics[-1]

            keys = ("_tr_loss", "_tr_acc", "_val_loss", "_val_acc")
            idx = 0

            for k in keys:    
                
                key = "t" + str(it) + k
                individual[key] = metrics[idx]
                idx += 1

        self.Record_Performance(individual)                


    def Record_Performance(self, individual):

        write_location = "{0}/{1}/Fitness_History_Seed{2}.csv".format(self.directory, self.folder, self.seed)

        bad_keys = ("network", "fitness", "ranked_fitness")
        sorted_keys = list(sorted([k for k in individual.keys() if k not in bad_keys]))

        # 't0_tr_acc', 't0_tr_loss', 't0_val_acc', 't0_val_loss', 't1_tr_acc', 't1_tr_loss', 't1_val_acc', 't1_val_loss'
        with open(write_location, "a+") as fitfile:
                    fitfile.write(",".join([str(individual[k]) for k in sorted_keys]))
                    fitfile.write("\n")


    def get_task(self, root, n_cl, n_inst, split='train'):
        if 'mnist' in root:
            return MNISTTask(root, n_cl, n_inst, split)
        elif 'omniglot' in root:
            return OmniglotTask(root, n_cl, n_inst, split)
        else:
            print('Unknown dataset')
        raise(Exception)

    def Save_Champion(self):

        write_location = "{0}/Champions/{1}".format(self.data_repo, self.seed)
        mkdir(write_location)     

        filename = "{0}/Champ{1}_G{2}_F{3}.p".format(write_location, self.seed, self.g, self.champion['fitness'])
        pickle_dict(self.champion, filename)

    def New_Swarm(self):

        for individual in self.population:
            for layer in self.means.keys():

                # Scale-location transformation of unit-normal: s*x + u
                dummy = np.random.normal(size=self.means[layer].shape)
            
                # Perform 'gradient ascent' on CURRENT mean for parameters in layer
                # Standard deviation is fixed to 1:
                dummy += (self.means[layer] + self.learning_rate*self.mean_grads[layer])
                dummy = torch.from_numpy(dummy)

                # Replace current members with new ones drawn from the updated distribution:
                individual['network'].state_dict()[layer].data.copy_(dummy)
                individual['fitness'] = 0

