import torch
from torchvision import datasets, transforms

import numpy as np
import scipy.ndimage.filters as filters
import utils.preproc as pre

batch_size = 100
test_batch_size = 10



#MNIST
MNIST_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=pre.MNIST_transform),
        batch_size=batch_size, shuffle=True)
MNIST_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=pre.MNIST_transform),
        batch_size=test_batch_size, shuffle=False)
MNIST_gmm_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=pre.MNIST_transform),
        batch_size=3000, shuffle=False)

X_MNIST = enumerate(MNIST_gmm_loader).__next__()[1][0].view(MNIST_gmm_loader.batch_size, 784)

#EMNIST
#EMNIST is rotated 90 degrees from MNIST
EMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=True, 
                    transform=pre.EMNIST_transform),
    batch_size=batch_size, shuffle=True)

EMNIST_gmm_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=True, 
                    transform=pre.EMNIST_transform),
    batch_size=3000, shuffle=False)

EMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=False, 
                    transform=pre.EMNIST_transform),
    batch_size=test_batch_size, shuffle=True)

EMNIST_test_loader_digits = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='digits', download=True, train=False, 
                    transform=pre.EMNIST_transform),
    batch_size=test_batch_size, shuffle=True)


X_EMNIST = enumerate(EMNIST_gmm_loader).__next__()[1][0].view(EMNIST_gmm_loader.batch_size, 784)


#FMNIST

FMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', download=True, train=True, 
                    transform=transforms.Compose([pre.MNIST_transform])),
    batch_size=batch_size, shuffle=True)

FMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', download=True, train=True, 
                    transform=transforms.Compose([pre.MNIST_transform])),
    batch_size=test_batch_size, shuffle=True)





GrayCIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=pre.gray_transform),
        batch_size=test_batch_size, shuffle=False)





Noise_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=pre.noise_transform),
        batch_size=test_batch_size, shuffle=False)





CIFAR10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=pre.MNIST_transform),
        batch_size=batch_size, shuffle=True)
CIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=pre.MNIST_transform),
        batch_size=test_batch_size, shuffle=False)
CIFAR10_gmm_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=pre.MNIST_transform),
        batch_size=3000, shuffle=False)

X_CIFAR10 = enumerate(CIFAR10_gmm_loader).__next__()[1][0].view(CIFAR10_gmm_loader.batch_size, 3072)

CIFAR100_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=pre.MNIST_transform, download=True),
        batch_size=10, shuffle=False)