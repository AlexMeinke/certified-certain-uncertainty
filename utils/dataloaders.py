import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import numpy as np
import scipy.ndimage.filters as filters
import utils.preproc as pre

batch_size = 100
test_batch_size = 10

download = False



#MNIST
MNIST_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=pre.MNIST_transform),
        batch_size=batch_size, shuffle=True)
MNIST_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=pre.MNIST_transform),
        batch_size=test_batch_size, shuffle=False)


#EMNIST
#EMNIST is rotated 90 degrees from MNIST
EMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', train=True, 
                    transform=pre.EMNIST_transform),
    batch_size=batch_size, shuffle=True)

EMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', train=False, 
                    transform=pre.EMNIST_transform),
    batch_size=test_batch_size, shuffle=True)


#FMNIST

FMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, 
                    transform=transforms.Compose([pre.MNIST_transform])),
    batch_size=batch_size, shuffle=True)

FMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, 
                    transform=transforms.Compose([pre.MNIST_transform])),
    batch_size=test_batch_size, shuffle=True)





GrayCIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=pre.gray_transform),
        batch_size=test_batch_size, shuffle=False)





Noise_train_loader_MNIST = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=pre.noise_transform),
        batch_size=batch_size, shuffle=False)

Noise_test_loader_MNIST = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=pre.noise_transform),
        batch_size=test_batch_size, shuffle=False)

Noise__train_loader_CIFAR10 = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, transform=pre.noise_transform),
        batch_size=batch_size, shuffle=False)

Noise__test_loader_CIFAR10 = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, transform=pre.noise_transform),
        batch_size=test_batch_size, shuffle=False)


CIFAR10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, transform=pre.MNIST_transform),
        batch_size=batch_size, shuffle=True)
CIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=pre.MNIST_transform),
        batch_size=test_batch_size, shuffle=False)


#CIFAR100
CIFAR100_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=pre.MNIST_transform),
        batch_size=test_batch_size, shuffle=False)

#SVHN
SVHN_test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data', split='train', transform=transforms.ToTensor(), download=download),
        batch_size=test_batch_size, shuffle=False)


def PrecomputeLoader(loader, batch_size=100, shuffle=True):
    X = []
    L = []
    for x,l in Noise_train_loader_MNIST:
        X.append(x)
        L.append(l)
    X = torch.cat(X, 0)
    L = torch.cat(L, 0)
    
    train = data_utils.TensorDataset(X, L)
    return data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)