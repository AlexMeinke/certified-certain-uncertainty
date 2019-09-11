import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

import utils.models as models
import utils.plotting as plotting
import utils.dataloaders as dl
import utils.traintest as tt
import utils.adversarial as adv
import utils.eval as ev
import utils.gmm_helpers as gmm_helpers
import model_params as params

import utils.mahalanobis as maha

import argparse


parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, FMNIST, SVHN, CIFAR10, CIFAR100')

hps = parser.parse_args()

dataset = hps.dataset


saving_string = dataset + '_base'
device = torch.device('cuda:' + str(hps.gpu))
#device = torch.device('cpu')


model_params = params.params_dict[dataset](augm_flag=True)


if dataset=='MNIST':
    model = maha.LeNet()
elif dataset=='FMNIST':
    model = maha.ResNet18(10, 1)
elif dataset in ['SVHN', 'CIFAR10']:
    model = maha.ResNet18(10, 3)
elif dataset=='CIFAR100':
    model = maha.ResNet18(100, 3)


model.to(device)


param_groups = [{'params':model.parameters(),'lr':model_params.lr, 'weight_decay':5e-4}]
    
if dataset=='MNIST':
    optimizer = optim.Adam(param_groups)
else: 
    optimizer = optim.SGD(param_groups, momentum=0.9)


for epoch in range(100):
    if epoch+1 in [50,75,90]:
        for group in optimizer.param_groups:
            group['lr'] *= .1
    
    #print(epoch)
    
    trainloss, correct_train = tt.training_dict['plain'](model, device,
                                                          model_params.train_loader,  
                                                          optimizer, epoch, 
                                                          verbose=-1)
    #print(str(epoch) + ': \t' + str(correct_train))

model = model.to('cpu')

torch.save(model, 'SavedModels/other/mahalanobis/' + saving_string+ '.pth')


final_model = maha.Mahalanobis(model.to(device), model_params, device)
final_model.cpu()

torch.save(final_model, 'SavedModels/other/mahalanobis/' + dataset + '.pth')