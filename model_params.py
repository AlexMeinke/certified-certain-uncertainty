import utils.models as models
import utils.dataloaders as dl
import resnet

class MNIST_params():
    def __init__(self, augm_flag=True):
        self.base_model = models.LeNetMadry()
        self.train_loader = dl.MNIST(train=True, augm_flag=augm_flag)
        self.test_loader = dl.MNIST(train=False)
        self.dim = 784
        self.loaders = [('FMNIST', dl.FMNIST(train=False)), 
             ('EMNIST', dl.EMNIST(train=False)),
             ('GrayCIFAR10', dl.GrayCIFAR10(train=False)),
             ('Noise', dl.Noise(dataset='MNIST'))]
        self.data_used = 60000

class SVHN_params():
    def __init__(self, augm_flag=True):
        self.base_model = resnet.ResNet50()
        self.train_loader = dl.SVHN(train=True, augm_flag=augm_flag)
        self.test_loader = dl.SVHN(train=False)
        self.dim = 3072
        self.loaders = [('CIFAR10', dl.CIFAR10(train=False)), 
             ('CIFAR100', dl.CIFAR100(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
             ('Noise', dl.Noise(dataset='SVHN'))]
        self.data_used = 50000
        
class CIFAR10_params():
    def __init__(self, augm_flag=True):
        self.base_model = resnet.ResNet50()
        self.train_loader = dl.CIFAR10(train=True, augm_flag=augm_flag)
        self.test_loader = dl.CIFAR10(train=False)
        self.dim = 3072
        self.loaders = [('CIFAR10', dl.SVHN(train=False)), 
             ('CIFAR100', dl.CIFAR100(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
             ('Noise', dl.Noise(dataset='SVHN'))]
        self.data_used = 50000
        
params_dict = {'MNIST':          MNIST_params,
               'SVHN':           SVHN_params,
               'CIFAR10':        CIFAR10_params,
              }