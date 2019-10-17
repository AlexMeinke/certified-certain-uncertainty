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

from importlib import reload

import utils.mc_dropout as mc

import argparse


parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, FMNIST, SVHN, CIFAR10')


hps = parser.parse_args()


dataset = hps.dataset

saving_string = dataset + '_base'
device = torch.device('cuda:' + str(hps.gpu))

model_params = params.params_dict[dataset](augm_flag=True)

def train(model, device, train_loader, optimizer, epoch, 
          verbose=100, noise_loader=None, epsilon=.3):
    
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        #output = F.log_softmax(model(data), dim=1)
        output = model(data)
        
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset)


if dataset=='MNIST':
    model = mc.LeNet()
elif dataset=='FMNIST':
    model = mc.vgg13(in_channels=1, num_classes=10)
elif dataset in ['SVHN', 'CIFAR10']:
    model = mc.vgg13(in_channels=3, num_classes=10)

model = model.to(device)

param_groups = [{'params':model.parameters(), 'lr':2e-4, 'weight_decay':0.}]
    
optimizer = optim.Adam(param_groups)


for epoch in range(100):
    if epoch+1 in [50,75,90]:
        for group in optimizer.param_groups:
            group['lr'] *= .1
 
    trainloss, correct_train = train(model, device, model_params.train_loader,  
                                     optimizer, epoch, verbose=-1)
    #print(str(epoch) + ': \t' + str(correct_train))

model = model.to('cpu')

mc_model = mc.MC_Model(model, iterations=10, classes=model_params.classes)

torch.save(mc_model, 'SavedModels/other/mcdo/' + dataset + '.pth')