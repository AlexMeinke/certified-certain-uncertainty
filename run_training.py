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

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')


parser.add_argument('--gpu', type=int, default=3, help='GPU index.')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate.')
parser.add_argument('--lam', type=float, default=-3., action='store', help='log of lambda.')
parser.add_argument('--n', type=int, default=1000, help='number of centroids.')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay for base model.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, SVHN, CIFAR10')
parser.add_argument('--use_gmm', type=bool, default=False, help='use gmm in training or not')
parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
parser.add_argument('--steps', type=int, default=40, help='PGD steps in training ACET')
parser.add_argument('--grad_vars', nargs='+', type=str, default=['mu','var','alpha'], help='variables in gmm that require grad')
parser.add_argument('--augm_flag', type=bool, default=False, help='whether to use data augmentation')
parser.add_argument('--gmm_path', type=str, default=None, help='path to gmm. If None, standard output of build_gmm.py is used.')
parser.add_argument('--verbose', type=bool, default=False, help='display training progress in command line')
parser.add_argument('--train_type', type=str, default='ACET', help='train on plain, CEDA or ACET')

hps = parser.parse_args()


model_params = params.params_dict[hps.dataset](augm_flag=hps.augm_flag)
base_model = model_params.base_model

loading_string = ('SavedModels/GMM/gmm_' + hps.dataset
                 +'_n' + str(hps.n)
                 +'_data_used' + str(model_params.data_used)
                 +'_augm_flag' + str(hps.augm_flag)
                 +'_alg_'+'scikit'
                 +'.pth') if hps.gmm_path is None else hps.gmm_path

if hps.use_gmm:
    saving_string = ('gmm' + hps.dataset
                     +'_lam' + str(hps.lam)
                     +'_n' + str(hps.n)
                     +'_lr' + str(hps.lr)
                     +'_augm_flag' + str(hps.augm_flag)
                     +'_train_type' + str(hps.train_type)
                     +'grad_vars')
    for name in hps.grad_vars:
        saving_string += ' ' + name
    if hps.gmm_path is not None:
        saving_string+= '_customGMM'
else:
    saving_string = ('base' + hps.dataset
                     +'_lr' + str(hps.lr)
                     +'_augm_flag' + str(hps.augm_flag)
                     +'_train_type' + str(hps.train_type))


device = torch.device('cuda:' + str(hps.gpu))
writer = SummaryWriter('runs/'+saving_string)

if hps.use_gmm:
    gmm = torch.load(loading_string) 
    
    gmm.alpha = nn.Parameter(gmm.alpha)
    gmm.mu.requires_grad = ('mu' in hps.grad_vars)
    gmm.logvar.requires_grad = ('var' in hps.grad_vars)
    gmm.alpha.requires_grad = ('alpha' in hps.grad_vars)
    model = models.RobustModel(base_model, gmm, hps.lam, dim=model_params.dim).to(device)
    model.loglam.requires_grad = False
else:
    model = base_model.to(device)

lr = hps.lr

if hps.use_gmm:
    param_groups = [{'params':model.mm.parameters(),'lr':lr, 'weight_decay':0.},
                    {'params':model.base_model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
else:
    param_groups = [{'params':model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
    
if hps.dataset=='MNIST':
    optimizer = optim.Adam(param_groups)
else: 
    optimizer = optim.SGD(param_groups, momentum=0.9)

for epoch in range(hps.epochs):
    if epoch+1 in [50,75,90]:
        for group in optimizer.param_groups:
            group['lr'] *= .1
    trainloss = tt.training_dict[hps.train_type](model, device, model_params.train_loader, 
                                  model_params.loaders[-1][1], 
                                  optimizer, epoch, 
                                  steps=hps.steps, verbose=hps.verbose)
    
    train_accuracy, train_ave_conf, train_loss = tt.test(model, device, model_params.train_loader )
    test_accuracy, test_ave_conf, test_loss = tt.test(model, device, model_params.test_loader )
    writer.add_scalar('InDistribution/TrainAccuracy', 100*train_accuracy, epoch)
    writer.add_scalar('InDistribution/TestAccuracy', 100*test_accuracy, epoch)
    
    writer.add_scalar('InDistribution/TrainMMC', train_ave_conf, epoch)
    writer.add_scalar('InDistribution/TestMMC', test_ave_conf, epoch)
    
    writer.add_scalar('InDistribution/TrainLoss', train_loss, epoch)
    writer.add_scalar('InDistribution/TestLoss', test_loss, epoch)
    if (epoch)%5==3:
        df = ev.evaluate(model, device, hps.dataset, model_params.loaders, writer=writer, epoch=epoch)

df = ev.evaluate(model, device, hps.dataset, model_params.loaders)
df.to_csv('results/gmm_model_joint'+saving_string+'.csv')


model = model.to('cpu')
torch.save(model, 'SavedModels/model_'+saving_string+ '.pth')
