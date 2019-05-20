import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import numpy as np
import scipy.ndimage.filters as filters
import utils.preproc as pre


train_batch_size = 128
test_batch_size = 100


def MNIST(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.MNIST('../data', train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=4)
    return loader


def EMNIST(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor(), pre.Transpose()] #EMNIST is rotated 90 degrees from MNIST
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.EMNIST('../data', split='letters', 
                              train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=1)
    return loader


def FMNIST(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.FashionMNIST('../data', train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=1)
    return loader


def GrayCIFAR10(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            pre.Gray()
                       ])]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
        ] + transform_base)
    
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.CIFAR10('../data', train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=1)
    return loader


def Noise(dataset, train=True, batch_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    pre.PermutationNoise(),
                    pre.GaussianFilter(),
                    pre.ContrastRescaling()
                    ])
    if dataset=='MNIST':
        dataset = datasets.MNIST('../data', train=train, transform=transform)
    elif dataset=='FMNIST':
        dataset = datasets.FashionMNIST('../data', train=train, transform=transform)
    elif dataset=='SVHN':
        dataset = datasets.SVHN('../data', split='train' if train else 'test', transform=transform)
    elif dataset=='CIFAR10':
        dataset = datasets.CIFAR10('../data', train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=4)
    loader = PrecomputeLoader(loader, batch_size=batch_size, shuffle=True)
    return loader


def CIFAR10(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)
    
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.CIFAR10('../data', train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=4)
    return loader


def CIFAR100(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)
    
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.CIFAR100('../data', train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=1)
    return loader


def SVHN(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
            
    if train:
        split = 'train'
    else:
        split = 'test'

    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)
    
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    
    transform = transform_train if (augm_flag and train) else transform_test
    
    dataset = datasets.SVHN('../data', split=split, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=train, num_workers=4)
    return loader


# LSUN classroom
def LSUN_CR(train=False, batch_size=None, augm_flag=False):
    if train:
        print('Warning: Training set for LSUN not available')
    if batch_size is None:
        batch_size=test_batch_size

    transform_base = [transforms.ToTensor()]
    transform = transforms.Compose([
            transforms.Resize(size=(32, 32))
        ] + transform_base)
    data_dir = '../data/LSUN'
    dataset = datasets.LSUN(data_dir, classes=['classroom_val'], transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=4)
    return loader


def ImageNetMinusCifar10(train=False, batch_size=None, augm_flag=False):
    if train:
        print('Warning: Training set for ImageNet not available')
    if batch_size is None:
        batch_size=test_batch_size
    dir_imagenet = '../data/imagenet/val/'
    n_test_imagenet = 30000

    transform = transforms.ToTensor()
    
    dataset = torch.utils.data.Subset(datasets.ImageFolder(dir_imagenet, transform=transform), 
                                            np.random.permutation(range(n_test_imagenet))[:10000])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                         shuffle=False, num_workers=1)
    return loader


def PrecomputeLoader(loader, batch_size=100, shuffle=True):
    X = []
    L = []
    for x,l in loader:
        X.append(x)
        L.append(l)
    X = torch.cat(X, 0)
    L = torch.cat(L, 0)
    
    train = data_utils.TensorDataset(X, L)
    return data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)


datasets_dict = {'MNIST':          MNIST,
                 'FMNIST':         FMNIST,
                 'cifar10_gray':   GrayCIFAR10,
                 'emnist':         EMNIST,
                 'CIFAR10':        CIFAR10,
                 'CIFAR100':       CIFAR100,
                 'SVHN':           SVHN,
                 'lsun_classroom': LSUN_CR,
                 'imagenet_minus_cifar10':  ImageNetMinusCifar10,
                 'noise': Noise,
                 }