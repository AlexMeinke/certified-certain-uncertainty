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

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--gpu', type=int, default=2, help='GPU index.')


hps = parser.parse_args()

device = torch.device('cuda:' + str(hps.gpu))

base_model = models.LeNetMadry().to(device)
lr = 1e-3

optimizer = optim.Adam(base_model.parameters(), lr=lr, weight_decay=5e-4)
for epoch in range(100):
    if epoch+1 in [50,75,90]:
        optimizer.param_groups[0]['lr'] *= .1
    tt.train_ACET(base_model, device, dl.MNIST_train_loader, dl.Noise_loader, optimizer, epoch)
torch.save(base_model, 'SavedModels/MNIST_base_model.pth')

base_df = ev.evaluate_MNIST(base_model, device)
base_df.to_csv('results/base_model')