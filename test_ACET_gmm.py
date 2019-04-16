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
import resnet

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpu', type=int, default=2, help='GPU index.')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate.')
parser.add_argument('--lam', type=float, default=-3., help='log of lambda.')
parser.add_argument('--n', type=int, default=1000, help='number of centroids.')
parser.add_argument('--decay', type=float, default=1000, help='weight decay for base model.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, CIFAR10')
parser.add_argument('--gmm', type=bool, default='True', help='use gmm in training or not')


hps = parser.parse_args()

device = torch.device('cuda:' + str(hps.gpu))


if hps.dataset=='MNIST':
    base_model = models.LeNetMadry().to(device)
    train_loader = dl.MNIST_train_loader
    noise_loader = dl.Noise_train_loader_MNIST
elif hps.dataset=='CIFAR10':
    base_model = resnet.ResNet50().to(device).to(device)
    train_loader = dl.CIFAR10_train_loader
    noise_loader = dl.Noise_train_loader_CIFAR10

if hps.gmm:
    loading_string = hps.dataset+'_n'+str(hps.n) 
    gmm = torch.load('SavedModels/gmm_'+loading_string+'.pth')
    model = models.RobustModel(base_model, gmm, hps.lam).to(device)
    model.loglam.requires_grad = False
else:
    model = base_model

saving_string = hps.dataset+'_lam'+str(hps.lam)+'_n'+str(hps.n)

lr = hps.lr

if hps.gmm:
    param_groups = [{'params':model.mm.parameters(),'lr':lr, 'weight_decay':0.},
                    {'params':model.base_model.parameters(),'lr':lr, 'weight_decay':hps.decay}]
else:
    param_groups = [{'params':model.parameters(),'lr':lr, 'weight_decay':hps.decay}]


optimizer = optim.Adam(param_groups)
for epoch in range(100):
    if epoch+1 in [50,75,90]:
        for group in optimizer.param_groups:
            group['lr'] *= .1
    tt.train_ACET(model, device, train_loader, noise_loader, optimizer, epoch)
torch.save(model, 'SavedModels/gmm_model_'+saving_string+ '.pth')


if hps.dataset=='MNIST':
    df = ev.evaluate_MNIST(model, device)
    df.to_csv('results/gmm_model_joint'+saving_string+'.csv')