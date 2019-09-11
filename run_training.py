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

from tensorboardX import SummaryWriter

import argparse


parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--lr', type=float, default=None, help='initial learning rate.')
parser.add_argument('--lr_gmm', type=float, default=1e-5, help='initial learning rate.')
parser.add_argument('--loglam', type=float, default=0., help='log of lambda.')
parser.add_argument('--n', type=int, default=100, help='number of centroids.')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay for base model.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, FMNIST, SVHN, CIFAR10, CIFAR100')
parser.add_argument('--use_gmm', type=bool, default=False, help='use gmm in training or not')
parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
parser.add_argument('--grad_vars', nargs='+', type=str, default=['mu', 'var'], 
                    help='variables in gmm that require grad')
parser.add_argument('--augm_flag', type=bool, default=False, help='whether to use data augmentation')
parser.add_argument('--gmm_path', type=str, default=None, 
                    help='path to gmm. If None, standard output of build_gmm.py is used.')
parser.add_argument('--verbose', type=int, default=-1, help='display training progress in command line')
parser.add_argument('--train_type', type=str, default='plain', help='train on plain, CEDA or CEDA_GMM')
parser.add_argument('--warmstart', type=str, default='None', help='warmstart base model on pretrained model')
parser.add_argument('--PCA', type=bool, default=False, help='use PCA in GMM metric')
parser.add_argument('--fit_out', type=bool, default=False, 
                    help='whether or not to fit out-distribution as well')
parser.add_argument('--steps', type=int, default=40, help='steps in PGD attack')

hps = parser.parse_args()


model_params = params.params_dict[hps.dataset](augm_flag=hps.augm_flag)

base_model = model_params.base_model
if hps.warmstart!='None':
    base_model = torch.load(hps.warmstart)
    
if hps.lr is None:
    hps.lr = model_params.lr
    
if hps.fit_out:
    hps.train_type = 'CEDA_GMM_OUT'


args = ''
args = (args + '_PCA') if hps.PCA else args

loading_string = ('SavedModels/GMM/gmm_' + hps.dataset
                 +'_n' + str(hps.n)
                 +'_data_used' + str(model_params.data_used)
                 +'_augm_flag' + str(hps.augm_flag)
                 + args
                 +'.pth') if hps.gmm_path is None else hps.gmm_path

loading_string_out = ('SavedModels/GMM/gmm_' + hps.dataset
                 +'_n' + str(hps.n)
                 +'_data_used' + str(model_params.data_used)
                 +'_augm_flag' + str(hps.augm_flag)
                 + args + '_OUT'
                 +'.pth')


if hps.use_gmm:
    gmm = torch.load(loading_string)
        
    saving_string = ('gmm_'+ args + hps.dataset
                     +'_lam' + str(hps.loglam)
                     +'_n' + str(hps.n)
                     +'_lr' + str(hps.lr)
                     +'_lrgmm' + str(hps.lr_gmm)
                     +'_augm_flag' + str(hps.augm_flag)
                     +'_train_type' + str(hps.train_type)
                     +'grad_vars')
    
    for name in hps.grad_vars:
        saving_string += ' ' + name
    if hps.gmm_path is not None:
        saving_string += '_customGMM'
    if hps.fit_out is not None:
        saving_string += '_OUT'
else:
    saving_string = ('base_' + hps.dataset
                     +'_lr' + str(hps.lr)
                     +'_augm_flag' + str(hps.augm_flag)
                     +'_train_type' + str(hps.train_type))


if hps.warmstart!='None':
    saving_string += '_warmstart'
if hps.train_type=='ACET':
    saving_string += '_steps' + str(hps.steps)

if torch.cuda.is_available():
    device = torch.device('cuda:' + str(hps.gpu))


if hps.use_gmm:
    gmm.alpha = nn.Parameter(gmm.alpha)
    gmm.mu.requires_grad = ('mu' in hps.grad_vars)
    gmm.logvar.requires_grad = ('var' in hps.grad_vars)
    gmm.alpha.requires_grad = ('alpha' in hps.grad_vars)
    
    if hps.fit_out:
        gmm_out = torch.load(loading_string_out)
        gmm_out.alpha = nn.Parameter(gmm.alpha)
        gmm_out.mu.requires_grad = ('mu' in hps.grad_vars)
        gmm_out.logvar.requires_grad = ('var' in hps.grad_vars)
        gmm_out.alpha.requires_grad = ('alpha' in hps.grad_vars)
        model = models.DoublyRobustModel(base_model, gmm, gmm_out,  
                                         hps.loglam, dim=model_params.dim, 
                                         classes=model_params.classes).to(device)
    else:
        model = models.RobustModel(base_model, gmm, hps.loglam, 
                                   dim=model_params.dim, 
                                   classes=model_params.classes).to(device)
    model.loglam.requires_grad = False
else:
    model = base_model.to(device)

lr = hps.lr
lr_gmm = hps.lr_gmm

if hps.use_gmm:
    param_groups = [{'params':model.mm.parameters(),'lr':lr_gmm, 'weight_decay':0.},
                    {'params':model.base_model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
    if hps.fit_out:
        param_groups = [{'params':model.mm.parameters(),'lr':lr_gmm, 'weight_decay':0.},
                        {'params':model.mm_out.parameters(),'lr':lr_gmm, 'weight_decay':0.},
                        {'params':model.base_model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
else:
    param_groups = [{'params':model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
    
    
if hps.dataset=='MNIST':
    optimizer = optim.Adam(param_groups)
else: 
    optimizer = optim.SGD(param_groups, momentum=0.9)


lam = model.loglam.data.exp().item() if hps.use_gmm else np.exp(hps.loglam)

writer = SummaryWriter('logs/'+saving_string+str(datetime.datetime.now()))

if hps.fit_out:
    noise_loader = dl.TinyImages(hps.dataset)
else:
    noise_loader = None


for epoch in range(hps.epochs):
    if epoch+1 in [50,75,90]:
        for group in optimizer.param_groups:
            group['lr'] *= .1
 
    losses = tt.training_dict[hps.train_type](model, device, model_params.train_loader,  
                                              optimizer, epoch, lam=lam, verbose=hps.verbose,
                                              epsilon=model_params.epsilon, noise_loader=noise_loader)
    
    trainloss, correct_train, likelihood_loss = losses

    correct, av_conf, test_loss = tt.test(model, device, model_params.test_loader)
    
    writer.add_scalar('InDistribution/TrainLoss', trainloss, epoch)
    writer.add_scalar('InDistribution/TrainAccuracy', correct_train, epoch)
    writer.add_scalar('InDistribution/likelihood_loss', likelihood_loss, epoch)
    
    writer.add_scalar('InDistribution/TestLoss', test_loss, epoch)
    writer.add_scalar('InDistribution/TestMMC', av_conf, epoch)
    writer.add_scalar('InDistribution/TestAccuracy', correct, epoch)
    writer.close()
    
    torch.save(model, 'Checkpoints/' + saving_string + '.pth')

model = model.to('cpu')
if hps.use_gmm:
    torch.save(model, 'SavedModels/' + saving_string + '.pth')
else:
    torch.save(model, 'SavedModels/base/' + saving_string + '.pth')
