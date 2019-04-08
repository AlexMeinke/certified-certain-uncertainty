import torch
from torchvision import datasets, transforms

import numpy as np
import scipy.ndimage.filters as filters

batch_size = 100
test_batch_size = 10


class Transpose(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return data.transpose(-1,-2)

transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

transform = transforms.ToTensor()

MNIST_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
MNIST_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False)
MNIST_gmm_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=3000, shuffle=False)

X_MNIST = enumerate(gmm_loader).__next__()[1][0].view(gmm_loader.batch_size, 784)

#EMNIST is rotated 90 degrees from MNIST
EMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=True, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=batch_size, shuffle=True)

EMNIST_gmm_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=True, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=3000, shuffle=False)

EMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=False, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=test_batch_size, shuffle=True)

EMNIST_test_loader_digits = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='digits', download=True, train=False, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=test_batch_size, shuffle=True)


X_EMNIST = enumerate(EMNIST_gmm_loader).__next__()[1][0].view(gmm_loader.batch_size, 784)




FMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', download=True, train=True, 
                    transform=transforms.Compose([transform])),
    batch_size=batch_size, shuffle=True)

FMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', download=True, train=True, 
                    transform=transforms.Compose([transform])),
    batch_size=test_batch_size, shuffle=True)





class Grey(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return data.mean(-3, keepdim=True)


grey_transform = transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            Grey()
                       ])

GreyCIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=grey_transform),
        batch_size=test_batch_size, shuffle=False)




class PermutationNoise(object):
    def __init__(self):
        pass
    def __call__(self, data):
        shape = data.shape
        new_data = 0*data
        for (i, x) in enumerate(data):
            idx = torch.tensor(np.random.permutation(np.prod(shape[1:])))
            new_data[i] = (x.view(np.prod(shape[1:]))[idx]).view(shape[1:])
        return new_data


class GaussianFilter(object):
    def __init__(self, sigma=1.):
        self.sigma = sigma
    def __call__(self, data):
        return filters.gaussian_filter(data, self.sigma)

noise_transform = transforms.Compose([
                            transforms.ToTensor(),
                            PermutationNoise(),
                            GaussianFilter()
                       ])

Noise_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=noise_transform),
        batch_size=test_batch_size, shuffle=False)





CIFAR10_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
CIFAR10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False)
CIFAR10_gmm_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform),
        batch_size=3000, shuffle=False)

X_CIFAR10 = enumerate(CIFAR10_gmm_loader).__next__()[1][0].view(gmm_loader.batch_size, 3072)

CIFAR100_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False, transform=transform, download=True),
        batch_size=10, shuffle=False)