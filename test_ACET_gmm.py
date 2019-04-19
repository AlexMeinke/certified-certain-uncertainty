import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from importlib import reload

import utils.models as models
import utils.plotting as plotting
import utils.dataloaders as dl
import utils.traintest as tt
import utils.adversarial as adv
import utils.eval as ev
import model_params as params
import resnet

from tensorboardX import SummaryWriter


import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpu', type=int, default=3, help='GPU index.')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate.')
parser.add_argument('--lam', type=float, default=-3., help='log of lambda.')
parser.add_argument('--n', type=int, default=1000, help='number of centroids.')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay for base model.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, SVHN, CIFAR10')
parser.add_argument('--use_gmm', type=bool, default=True, help='use gmm in training or not')
parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
parser.add_argument('--steps', type=int, default=40, help='PGD steps in training ACET')
parser.add_argument('--grad_vars', nargs='+', type=str, default=['mu','var','alpha'], help='variables in gmm that require grad')
parser.add_argument('--augm_flag', type=bool, default=True, help='whether to use data augmentation')

hps = parser.parse_args()

device = torch.device('cuda:' + str(hps.gpu))
writer = SummaryWriter()

model_params = params.params_dict[hps.dataset](augm_flag=hps.augm_flag)
base_model = model_params.base_model



if hps.use_gmm:
    loading_string = hps.dataset+'_n'+str(hps.n) 
    gmm = torch.load('SavedModels/gmm_'+loading_string+'.pth')
    
    gmm.alpha = nn.Parameter(gmm.alpha)
    gmm.mu.requires_grad = ('mu' in hps.grad_vars)
    gmm.logvar.requires_grad = ('var' in hps.grad_vars)
    gmm.alpha.requires_grad = ('alpha' in hps.grad_vars)
    model = models.RobustModel(base_model, gmm, hps.lam, dim=model_params.dim).to(device)
    model.loglam.requires_grad = False
else:
    model = base_model.to(device)

saving_string = hps.dataset+'_lam'+str(hps.lam)+'_n'+str(hps.n)

lr = hps.lr

if hps.use_gmm:
    param_groups = [{'params':model.mm.parameters(),'lr':lr, 'weight_decay':0.},
                    {'params':model.base_model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
else:
    param_groups = [{'params':model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
    
optimizer = optim.Adam(param_groups)


for epoch in range(hps.epochs):
    if epoch+1 in [50,75,90]:
        for group in optimizer.param_groups:
            group['lr'] *= .1
    trainloss = tt.train_ACET(model, device, model_params.train_loader, 
                              model_params.loaders[-1][1], 
                              optimizer, epoch, 
                              steps=hps.steps, verbose=False)
    
    correct, ave_conf = tt.test(model, device, model_params.test_loader )
    writer.add_scalar('TestSet/TrainLoss', trainloss, epoch)
    writer.add_scalar('TestSet/Correct', correct, epoch)
    writer.add_scalar('TestSet/Confidence', ave_conf, epoch)
    if (epoch)%5==0:
        df = ev.evaluate(model, device, hps.dataset, model_params.loaders, writer=writer, epoch=epoch)


if use_gmm:       
    torch.save(model, 'SavedModels/gmm_model_'+saving_string+ '.pth')
else:
    torch.save(model, 'SavedModels/base_model_'+str(hps.dataset)+ '.pth')


if hps.dataset=='MNIST':
    df = ev.evaluate(model, device, dataset=hps.dataset)
    df.to_csv('results/gmm_model_joint'+saving_string+'.csv')
