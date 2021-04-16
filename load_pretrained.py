import torch
import utils.models as models
import model_params

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, FMNIST, SVHN, CIFAR10, CIFAR100')
parser.add_argument('--pretrained', type=str, required=True, help='give path to pretrained model weights')
hps = parser.parse_args()

params = model_params.params_dict[hps.dataset]()

n = 100
dim = params.dim
classes = params.classes

metric = models.PCAMetric(torch.rand(params.dim, params.dim))
gmm = models.GMM(n, dim, metric=metric)
gmm_out = models.GMM(n, dim, metric=metric)
base_model = params.base_model

model = models.DoublyRobustModel(base_model, gmm, gmm_out,  
                                 loglam=0., dim=dim, 
                                 classes=classes)

state_dict = torch.load(hps.pretrained)
model.load_state_dict(state_dict)

torch.save(model, hps.pretrained + 'h')