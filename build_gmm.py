import torch
import utils.models as models
import utils.dataloaders as dl

import argparse

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--n', type=int, default=1000, help='number of Gaussians.')

hps = parser.parse_args()

gmm = models.GMM(hps.n, 784)

X = []
for x, f in dl.MNIST_train_loader:
    X.append(x.view(-1,784))
X = torch.cat(X, 0)

gmm.find_solution(X, initialize=True, iterate=True, use_kmeans=False)

torch.save(gmm, 'SavedModels/gmm_MNIST_n'+str(hps.n)+'.pth')
print('Found gmm for MNIST')