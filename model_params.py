'''
    In this file one can set the hyperparameters for training, which model architectures to use
    as well as include or exclude out-distribution datasets that one wishes to test on
    Nearly all scripts reference this file
'''


import utils.models as models
import utils.dataloaders as dl
import utils.resnet_orig as resnet


class MNIST_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.data_name = 'MNIST'
        self.dim = 784
        self.base_model = models.LeNetMadry()
        
        self.train_loader = dl.MNIST(train=True, batch_size=batch_size, augm_flag=augm_flag)
        self.cali_loader = dl.MNIST(train=True, batch_size=batch_size, augm_flag=False)
        self.test_loader = dl.MNIST(train=False)

        self.loaders = [('FMNIST', dl.FMNIST(train=False)), 
             ('EMNIST', dl.EMNIST(train=False)),
             ('GrayCIFAR10', dl.GrayCIFAR10(train=False)),
        #     ('TinyImages', dl.TinyImages(self.data_name, batch_size=batch_size, train=False)),
             ('Noise', dl.Noise(dataset=self.data_name, batch_size=batch_size)),
             ('UniformNoise', dl.UniformNoise(self.data_name, batch_size=batch_size))]
        
        self.tinyimage_loader = dl.TinyImages(self.data_name, batch_size=100)
        
        self.data_used = 60000
        self.epsilon = 0.3
        self.lr = 1e-3
        self.classes = 10
        
        
class FMNIST_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.data_name = 'FMNIST'
        
        self.base_model = resnet.ResNet18(num_of_channels=1)
        
        self.train_loader = dl.FMNIST(train=True, batch_size=batch_size, augm_flag=augm_flag)
        self.cali_loader = dl.FMNIST(train=True, batch_size=batch_size, augm_flag=False)
        self.test_loader = dl.FMNIST(train=False, augm_flag=False)
        self.dim = 784
        self.loaders = [('MNIST', dl.MNIST(train=False)), 
             ('EMNIST', dl.EMNIST(train=False)),
             ('GrayCIFAR10', dl.GrayCIFAR10(train=False)),
        #     ('TinyImages', dl.TinyImages(self.data_name, batch_size=batch_size, train=False)),
             ('Noise', dl.Noise(dataset=self.data_name, batch_size=batch_size)),
             ('UniformNoise', dl.UniformNoise(dataset=self.data_name, batch_size=batch_size))]
        
        self.tinyimage_loader = dl.TinyImages(self.data_name, batch_size=100)
        
        self.data_used = 60000
        self.epsilon = 0.3
        self.lr = 0.1
        self.classes = 10

        
class SVHN_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.data_name = 'SVHN'
        self.base_model = resnet.ResNet18()
        self.train_loader = dl.SVHN(train=True, batch_size=batch_size, augm_flag=augm_flag)
        self.cali_loader = dl.SVHN(train=True, batch_size=batch_size, augm_flag=False)
        self.test_loader = dl.SVHN(train=False)
        self.dim = 3072
        self.loaders = [('CIFAR10', dl.CIFAR10(train=False)), 
             ('CIFAR100', dl.CIFAR100(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
        #     ('TinyImages', dl.TinyImages(self.data_name, batch_size=batch_size, train=False)),
             ('Noise', dl.Noise(dataset='SVHN', batch_size=batch_size)),
             ('UniformNoise', dl.UniformNoise(dataset=self.data_name, batch_size=batch_size))]
        
        self.tinyimage_loader = dl.TinyImages(self.data_name, batch_size=100)
        
        self.data_used = 50000
        self.epsilon = 0.3
        self.lr = 0.1
        self.classes = 10
        
        
class CIFAR10_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.data_name = 'CIFAR10'
        self.base_model = resnet.ResNet18()
        self.train_loader = dl.CIFAR10(train=True, batch_size=batch_size, augm_flag=augm_flag)
        self.cali_loader = dl.CIFAR10(train=True, batch_size=batch_size, augm_flag=False)
        self.test_loader = dl.CIFAR10(train=False)
        self.dim = 3072
        self.loaders = [('SVHN', dl.SVHN(train=False)), 
             ('CIFAR100', dl.CIFAR100(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
        #     ('TinyImages', dl.TinyImages(self.data_name, batch_size=batch_size, train=False)),
             ('Noise', dl.Noise(dataset='CIFAR10', batch_size=batch_size)),
             ('UniformNoise', dl.UniformNoise(dataset=self.data_name, batch_size=batch_size))]
        
        self.tinyimage_loader = dl.TinyImages(self.data_name, batch_size=100)
        
        self.data_used = 50000
        self.epsilon = 0.3
        self.lr = 0.1
        self.classes = 10
        
        
class CIFAR100_params():
    def __init__(self, augm_flag=True, batch_size=128):
        self.data_name = 'CIFAR100'
        self.base_model = resnet.ResNet18(num_classes=100)
        self.train_loader = dl.CIFAR100(train=True, batch_size=batch_size, augm_flag=augm_flag)
        self.cali_loader = dl.CIFAR100(train=True, batch_size=batch_size, augm_flag=False)
        self.test_loader = dl.CIFAR100(train=False)
        self.dim = 3072
        self.loaders = [('SVHN', dl.SVHN(train=False)), 
             ('CIFAR10', dl.CIFAR10(train=False)),
             ('LSUN_CR', dl.LSUN_CR(train=False)),
             ('Imagenet-',dl.ImageNetMinusCifar10(train=False)),
        #     ('TinyImages', dl.TinyImages(self.data_name, batch_size=batch_size, train=False)),
             ('Noise', dl.Noise(dataset='CIFAR100', batch_size=batch_size)),
             ('UniformNoise', dl.UniformNoise(dataset=self.data_name, batch_size=batch_size))]
        
        self.tinyimage_loader = dl.TinyImages(self.data_name, batch_size=100)
        
        
        self.data_used = 50000
        self.epsilon = 0.3
        self.lr = 0.1
        self.classes = 100
       
    
params_dict = {'MNIST':          MNIST_params,
               'FMNIST':         FMNIST_params,
               'SVHN':           SVHN_params,
               'CIFAR10':        CIFAR10_params,
               'CIFAR100':       CIFAR100_params,
              }