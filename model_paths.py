'''
    Specify model paths of all models that you wish to benchmark using either 
    gen_eval.py or gen_attack_stats.py
'''


import collections


class MNIST_models():
    def __init__(self):
        file_base = ('SavedModels/base/'
                     'base_MNIST_lr0.001_augm_flagTrue_train_typeplain.pth'
                    )
        
        file_MCDO = ('SavedModels/other/mcdo/MNIST.pth')
        
        file_GAN = ('SavedModels/other/gan/MNIST.pth')
        
        file_ODIN = ('SavedModels/other/odin/'
                     'MNIST_OUTSEEDS_ODIN.pth'
                    )
        
        file_single_maha = ('SavedModels/other/single_mahalanobis/MNIST.pth')
        
        file_ACET_OUT = ('SavedModels/base/'
                     'base_MNIST_lr0.001_augm_flagTrue_train_typeACET_OUT_steps40.pth'
                    )
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_MNIST.pth')
        
        file_CCU_double = ('SavedModels/'
                         'gmm__PCAMNIST_lam0.0_n100_lr0.001_lrgmm1e-05_'
                         'augm_flagTrue_train_typeCEDA_GMM_OUTgrad_vars mu var_OUT.pth')
        
        file_CCU_single = ('SavedModels/'
                      'gmm__PCAMNIST_lam0.0_n100_lr0.001_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        
        files = [file_base, file_MCDO, file_GAN,
                 file_ODIN, file_single_maha, file_ACET_OUT,
                 file_hendrycks, file_CCU_double, file_CCU_single]
        
        keys = ['Base', 'MCD', 'GAN', 'ODIN', 'Maha', 'ACET', 'OE', 'CCU', 'CCU']
               
        
        self.file_dict = collections.OrderedDict(zip(keys, files))

        
class FMNIST_models():
    def __init__(self):
        file_base = ('SavedModels/base/'
                     'base_FMNIST_lr0.001_augm_flagTrue_train_typeplain.pth'
                    )
        
        file_MCDO = ('SavedModels/other/mcdo/FMNIST.pth')
        
        file_GAN = ('SavedModels/other/gan/FMNIST.pth')
        
        file_ODIN = ('SavedModels/other/odin/'
                     'FMNIST_OUTSEEDS_ODIN.pth'
                    )
        
        file_single_maha = ('SavedModels/other/single_mahalanobis/FMNIST.pth')
        
        file_ACET_OUT = ('SavedModels/base/'
                     'base_FMNIST_lr0.1_augm_flagTrue_train_typeACET_OUT_steps40.pth'
                    )
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_FMNIST.pth')
        
        file_CCU_double = ('SavedModels/'
                         'gmm__PCAFMNIST_lam0.0_n100_lr0.1_lrgmm1e-05_'
                         'augm_flagTrue_train_typeCEDA_GMM_OUTgrad_vars mu var_OUT.pth')
        
        file_CCU_single = ('SavedModels/'
                      'gmm__PCAFMNIST_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        
        files = [file_base, file_MCDO, file_GAN,
                 file_ODIN, file_single_maha, file_ACET_OUT,
                 file_hendrycks, file_CCU_double, file_CCU_single]
        
        keys = ['Base', 'MCD', 'GAN', 'ODIN', 'Maha', 'ACET', 'OE', 'CCU', 'CCU']
        
        self.file_dict = collections.OrderedDict(zip(keys, files))
       
    
class SVHN_models():
    def __init__(self):
        file_base = ('SavedModels/base/'
                     'base_SVHN_lr0.001_augm_flagTrue_train_typeplain.pth'
                    )
        
        file_MCDO = ('SavedModels/other/mcdo/SVHN.pth')
        
        file_GAN = ('SavedModels/other/gan/SVHN.pth')
        
        file_ODIN = ('SavedModels/other/odin/'
                     'SVHN_OUTSEEDS_ODIN.pth'
                    )
        
        file_single_maha = ('SavedModels/other/single_mahalanobis/SVHN.pth')
        
        file_ACET_OUT = ('SavedModels/base/'
                     'base_SVHN_lr0.1_augm_flagTrue_train_typeACET_OUT_steps40.pth'
                    )
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_SVHN.pth')
        
        file_CCU_double = ('SavedModels/'
                         'gmm__PCASVHN_lam0.0_n100_lr0.1_lrgmm1e-05_'
                         'augm_flagTrue_train_typeCEDA_GMM_OUTgrad_vars mu var_OUT.pth')
        
        file_CCU_single = ('SavedModels/'
                      'gmm__PCASVHN_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        
        files = [file_base, file_MCDO, file_GAN,
                 file_ODIN, file_single_maha, file_ACET_OUT,
                 file_hendrycks, file_CCU_double, file_CCU_single]
        
        keys = ['Base', 'MCD', 'GAN', 'ODIN', 'Maha', 'ACET', 'OE', 'CCU', 'CCU']
        
        self.file_dict = collections.OrderedDict(zip(keys, files))


class CIFAR10_models():
    def __init__(self):
        file_base = ('SavedModels/base/'
                     'base_CIFAR10_lr0.001_augm_flagTrue_train_typeplain.pth'
                    )
        
        file_MCDO = ('SavedModels/other/mcdo/CIFAR10.pth')
        
        file_GAN = ('SavedModels/other/gan/CIFAR10.pth')
        
        file_ODIN = ('SavedModels/other/odin/'
                     'CIFAR10_OUTSEEDS_ODIN.pth'
                    )
        
        file_single_maha = ('SavedModels/other/single_mahalanobis/CIFAR10.pth')
        
        file_ACET_OUT = ('SavedModels/base/'
                     'base_CIFAR10_lr0.1_augm_flagTrue_train_typeACET_OUT_steps40.pth'
                    )
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_CIFAR10.pth')
        
        file_CCU_double = ('SavedModels/'
                         'gmm__PCACIFAR10_lam0.0_n100_lr0.1_lrgmm1e-05_'
                         'augm_flagTrue_train_typeCEDA_GMM_OUTgrad_vars mu var_OUT.pth')
        
        file_CCU_single = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        
        files = [file_base, file_MCDO, file_GAN,
                 file_ODIN, file_single_maha, file_ACET_OUT,
                 file_hendrycks, file_CCU_double, file_CCU_single]
        
        keys = ['Base', 'MCD', 'GAN', 'ODIN', 'Maha', 'ACET', 'OE', 'CCU', 'CCU']
        
        self.file_dict = collections.OrderedDict(zip(keys, files))
        

model_dict =  {'MNIST':          MNIST_models,
               'FMNIST':         FMNIST_models,
               'SVHN':           SVHN_models,
               'CIFAR10':        CIFAR10_models,
              }