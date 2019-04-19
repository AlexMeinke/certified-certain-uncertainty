import torch
import utils.models as models
import utils.dataloaders as dl

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--n', type=int, default=1000, help='number of Gaussians.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, SVHN, CIFAR10')
parser.add_argument('--verbose', type=bool, default=False, help='whether to print current iteration')

hps = parser.parse_args()

if hps.dataset=='MNIST':
    dim = 784
    loader = dl.MNIST(train=True,augm_flag=False)
elif hps.dataset=='SVHN':
    dim = 3072
    loader = dl.SVHN(train=True,augm_flag=False)
elif hps.dataset=='CIFAR10':
    dim = 3072
    loader = dl.CIFAR10(train=True,augm_flag=False)

gmm = models.GMM(hps.n, dim)

X = []
for x, f in loader:
    X.append(x.view(-1,dim))
X = torch.cat(X, 0)
    
if hps.dataset=='SVHN':
    X = X[:50000] #needed to keep memory of distance matrix below 800 GB

gmm.find_solution(X, initialize=True, iterate=True, use_kmeans=False, verbose=verbose)
gmm.alpha = nn.Parameter(gmm.alpha)

torch.save(gmm, 'SavedModels/gmm_'+hps.dataset+'_n'+str(hps.n)+'.pth')
print('Done')