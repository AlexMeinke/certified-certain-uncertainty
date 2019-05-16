import utils.models as models
import utils.dataloaders as dl
import utils.resnet_orig as resnet


class MNIST_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.base_model = models.LeNetMadry()
        self.train_loader = dl.MNIST(train=True, batch_size=128, augm_flag=augm_flag)
        self.cali_loader = dl.MNIST(train=True, batch_size=128, augm_flag=False)
        self.test_loader = dl.MNIST(train=False)
        self.dim = 784
        self.loaders = [('FMNIST', dl.FMNIST(train=False)), 
             ('EMNIST', dl.EMNIST(train=False)),
             ('GrayCIFAR10', dl.GrayCIFAR10(train=False)),
             ('Noise', dl.Noise(dataset='MNIST', batch_size=batch_size))]
        self.data_used = 60000
        self.epsilon = 0.3
        self.lr = 1e-3
        
        
class FMNIST_params():
    def __init__(self, augm_flag=True, batch_size=128):
        # self.base_model = models.LeNetMadry()
        self.base_model = resnet.ResNet18Gray()
        self.train_loader = dl.FMNIST(train=True, batch_size=128, augm_flag=augm_flag)
        self.cali_loader = dl.FMNIST(train=True, batch_size=128, augm_flag=False)
        self.test_loader = dl.FMNIST(train=False, augm_flag=False)
        self.dim = 784
        self.loaders = [('MNIST', dl.MNIST(train=False)), 
             ('EMNIST', dl.EMNIST(train=False)),
             ('GrayCIFAR10', dl.GrayCIFAR10(train=False)),
             ('Noise', dl.Noise(dataset='FMNIST', batch_size=batch_size))]
        self.data_used = 60000
        self.epsilon = 0.3
        self.lr = 0.1

        
class SVHN_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.base_model = resnet.ResNet18()
        self.train_loader = dl.SVHN(train=True, batch_size=batch_size, augm_flag=augm_flag)
        self.cali_loader = dl.SVHN(train=True, batch_size=batch_size, augm_flag=False)
        self.test_loader = dl.SVHN(train=False)
        self.dim = 3072
        self.loaders = [('CIFAR10', dl.CIFAR10(train=False)), 
             ('CIFAR100', dl.CIFAR100(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
             ('Noise', dl.Noise(dataset='SVHN', batch_size=batch_size))]
        self.data_used = 50000
        self.epsilon = 0.1
        self.lr = 0.1
        
        
class CIFAR10_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.base_model = resnet.ResNet18()
        self.train_loader = dl.CIFAR10(train=True, batch_size=128, augm_flag=augm_flag)
        self.cali_loader = dl.CIFAR10(train=True, batch_size=128, augm_flag=False)
        self.test_loader = dl.CIFAR10(train=False)
        self.dim = 3072
        self.loaders = [('SVHN', dl.SVHN(train=False)), 
             ('CIFAR100', dl.CIFAR100(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
             ('Noise', dl.Noise(dataset='CIFAR10', batch_size=batch_size))]
        self.data_used = 50000
        self.epsilon = 0.1
        self.lr = 0.1
       
    
params_dict = {'MNIST':          MNIST_params,
               'FMNIST':         FMNIST_params,
               'SVHN':           SVHN_params,
               'CIFAR10':        CIFAR10_params,
              }