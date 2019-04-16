import torch
import utils.models as models
import utils.dataloaders as dl

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--n', type=int, default=1000, help='number of Gaussians.')
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST, CIFAR10')

hps = parser.parse_args()

if hps.dataset=='MNIST':
    dim = 784
    loader = dl.MNIST_train_loader
elif hps.dataset=='CIFAR10':
    dim = 3072
    loader = dl.CIFAR10_train_loader

gmm = models.GMM(hps.n, dim)

X = []
for x, f in loader:
    X.append(x.view(-1,dim))
X = torch.cat(X, 0)

gmm.find_solution(X, initialize=True, iterate=True, use_kmeans=False)

torch.save(gmm, 'SavedModels/gmm_'+hps.dataset+'_n'+str(hps.n)+'.pth')
print('Done')