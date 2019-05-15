import torch
import torch.nn as nn
import utils.models as models
import utils.dataloaders as dl
from sklearn import mixture
import numpy as np
import utils.gmm_helpers as gmm_helpers

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--n', type=int, default=1000, help='number of Gaussians.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, SVHN, CIFAR10.')
parser.add_argument('--verbose', type=bool, default=False, help='whether to print current iteration.')
parser.add_argument('--data_used', type=int, default=None, help='number of datapoints to be used.')
parser.add_argument('--alg', type=str, default='scikit', help='which algorithm to use, [EM, scikit, EM-kmeans].')
parser.add_argument('--augm_flag', type=bool, default=False, help='whether to use data augmentation.')
parser.add_argument('--percentile', type=float, default=1., help='percentile for rescaling.')
parser.add_argument('--PCA', type=bool, default=False, help='initialize for using in PCA metric.')

hps = parser.parse_args()

if hps.dataset=='MNIST':
    dim = 784
    loader = dl.MNIST(train=True,augm_flag=hps.augm_flag)
    hps.data_used = 60000 if hps.data_used is None else hps.data_used
if hps.dataset=='FMNIST':
    dim = 784
    loader = dl.FMNIST(train=True,augm_flag=hps.augm_flag)
    hps.data_used = 60000 if hps.data_used is None else hps.data_used
elif hps.dataset=='SVHN':
    dim = 3072
    loader = dl.SVHN(train=True,augm_flag=hps.augm_flag)
    hps.data_used = 50000 if hps.data_used is None else hps.data_used
elif hps.dataset=='CIFAR10':
    dim = 3072
    loader = dl.CIFAR10(train=True,augm_flag=hps.augm_flag)
    hps.data_used = 50000 if hps.data_used is None else hps.data_used


X = []
for x, f in loader:
    X.append(x.view(-1,dim))
X = torch.cat(X, 0)

X = X[:hps.data_used] #needed to keep memory of distance matrix below 800 GB

if hps.PCA:
    metric = models.PCAMetric( X, p=2, min_sv_factor=10000)
    X = ( (X@metric.comp_vecs.t()) / metric.singular_values[None,:] )
else:
    metric = models.LpMetric()

gmm = models.GMM(hps.n, dim, metric=metric)
if hps.alg=='EM':
    gmm.find_solution(X, initialize=True, iterate=True, use_kmeans=False, verbose=hps.verbose)
    gmm.alpha = nn.Parameter(gmm.alpha)
elif hps.alg=='EM-kmeans':
    gmm.find_solution(X, initialize=True, iterate=True, use_kmeans=True, verbose=hps.verbose)
    gmm.alpha = nn.Parameter(gmm.alpha)
elif hps.alg=='scikit':
    clf = mixture.GaussianMixture(n_components=hps.n, 
                                  covariance_type='spherical',
                                  max_iter=500)
    clf.fit(X)
    mu = torch.tensor(clf.means_ ,dtype=torch.float)
    logvar = torch.tensor(np.log(clf.covariances_) ,dtype=torch.float)
    alpha = torch.tensor(np.log(clf.weights_) ,dtype=torch.float)
    gmm = models.GMM(hps.n, dim, mu=mu, logvar=logvar, alpha=alpha, metric=metric)
else:    
    raise ValueError("Invalid algorithm "+ str(hps.alg))
    
if hps.PCA:
    gmm.mu.data = ( (gmm.mu.data * metric.singular_values[None,:] ) @ metric.comp_vecs.t().inverse() )
    
saving_string = ('SavedModels/GMM/gmm_' + hps.dataset
                 +'_n' + str(hps.n)
                 +'_data_used' + str(hps.data_used)
                 +'_augm_flag' + str(hps.augm_flag)
                 +'_alg_'+str(hps.alg))

if hps.PCA:
    saving_string += '_PCA'


torch.save(gmm, saving_string + '.pth')

gmm = gmm_helpers.rescale(gmm, 1., loader)
saving_string += '_rescaled'
torch.save(gmm, saving_string+'.pth')
print('Done')
