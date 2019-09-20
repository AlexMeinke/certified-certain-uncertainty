import torch
import torch.nn as nn
import utils.models as models
import utils.dataloaders as dl
from sklearn import mixture
import numpy as np
import utils.gmm_helpers as gmm_helpers
import model_params
import torchvision.transforms as trn

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
#parser.add_argument('--n', type=int, default=100, help='number of Gaussians.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, SVHN, CIFAR10.')
parser.add_argument('--data_used', type=int, default=None, help='number of datapoints to be used.')
parser.add_argument('--augm_flag', type=bool, default=False, help='whether to use data augmentation.')
parser.add_argument('--PCA', type=bool, default=False, help='initialize for using in PCA metric.')
parser.add_argument('--n', nargs='+', type=int, default=[100])

hps = parser.parse_args()


params = model_params.params_dict[hps.dataset](augm_flag=hps.augm_flag)

dim = params.dim
loader = params.train_loader
hps.data_used = params.data_used if hps.data_used is None else hps.data_used

"""
X = []
for x, f in loader:
    X.append(x.view(-1,dim))
X = torch.cat(X, 0)

X = X[:hps.data_used] #needed to keep memory of distance matrix below 800 GB


if hps.PCA:
    metric = models.PCAMetric( X, p=2, min_sv_factor=1e6)
    X = ( (X@metric.comp_vecs.t()) / metric.singular_values_sqrt[None,:] )
else:
    metric = models.LpMetric()

for n in hps.n:
    print(n)
    gmm = models.GMM(n, dim, metric=metric)

    clf = mixture.GMM(n_components=n, covariance_type='spherical', params='mc')
    
    clf.fit(X)
    mu = torch.tensor(clf.means_ ,dtype=torch.float)
    
    logvar = torch.tensor(np.log(clf.covars_[:,0]) ,dtype=torch.float)
    logvar = 0.*logvar + logvar.exp().mean().log()
    
    alpha = torch.tensor(np.log(clf.weights_) ,dtype=torch.float)
    gmm = models.GMM(n, dim, mu=mu, logvar=logvar, metric=metric)


    if hps.PCA:
        gmm.mu.data = ( (gmm.mu.data * metric.singular_values_sqrt[None,:] ) 
                       @ metric.comp_vecs.t().inverse() )

    saving_string = ('SavedModels/GMM/gmm_' + hps.dataset
                     +'_n' + str(n)
                     +'_data_used' + str(hps.data_used)
                     +'_augm_flag' + str(hps.augm_flag))

    if hps.PCA:
        saving_string += '_PCA'


    torch.save(gmm, saving_string + '.pth')
"""

### Only temporary!!!
saving_string = ('SavedModels/GMM/gmm_' + hps.dataset
                 +'_n' + str(hps.n[0])
                 +'_data_used' + str(hps.data_used)
                 +'_augm_flag' + str(hps.augm_flag) + '_PCA.pth')

gmm = torch.load(saving_string)
metric = gmm.metric
### Only temporary!!!
    
out_loader = dl.TinyImages(hps.dataset)

    
X = []
for idx, (x, f) in enumerate(out_loader):
    if idx>100:
        break;
    X.append(x.view(-1,dim))
X = torch.cat(X, 0)


if hps.PCA:
    X = ( (X@metric.comp_vecs.t()) / metric.singular_values_sqrt[None,:] ) 

    
for n in hps.n:
    print(n)
    # Out GMM
    gmm = models.GMM(n, dim, metric=metric)

    clf = mixture.GMM(n_components=n, covariance_type='spherical', params='mc')
    
    clf.fit(X)
    mu = torch.tensor(clf.means_ ,dtype=torch.float)
    
    logvar = torch.tensor(np.log(clf.covars_[:,0]) ,dtype=torch.float)
    logvar = 0.*logvar + logvar.exp().mean().log()
    
    alpha = torch.tensor(np.log(clf.weights_) ,dtype=torch.float)
    gmm = models.GMM(n, dim, mu=mu, logvar=logvar, metric=metric)


    if hps.PCA:
        gmm.mu.data = ( (gmm.mu.data * metric.singular_values_sqrt[None,:] ) 
                       @ metric.comp_vecs.t().inverse() )

    saving_string = ('SavedModels/GMM/gmm_' + hps.dataset
                     +'_n' + str(n)
                     +'_data_used' + str(hps.data_used)
                     +'_augm_flag' + str(hps.augm_flag))

    if hps.PCA:
        saving_string += '_PCA'


    torch.save(gmm, saving_string + '_OUT' + '.pth')

print('Done')
