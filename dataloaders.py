import torch
from torchvision import datasets, transforms

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

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False)
gmm_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=3000, shuffle=True)


#EMNIST is rotated 90 degrees from MNIST
EMNIST_train_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=True, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=batch_size, shuffle=True)

EMNIST_test_loader = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='letters', download=True, train=False, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=test_batch_size, shuffle=True)

EMNIST_test_loader_digits = torch.utils.data.DataLoader(
    datasets.EMNIST('../data', split='digits', download=True, train=False, 
                    transform=transforms.Compose([transform, Transpose()])),
    batch_size=test_batch_size, shuffle=True)

fashion_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', download=True, train=True, 
                    transform=transforms.Compose([transform])),
    batch_size=batch_size, shuffle=True)

X_MNIST = enumerate(gmm_loader).__next__()[1][0].view(gmm_loader.batch_size, 784)