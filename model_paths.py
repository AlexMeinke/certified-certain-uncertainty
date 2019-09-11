import collections


class MNIST_models():
    def __init__(self):
        file_joint = ('SavedModels/'
                      'gmm__PCAMNIST_lam0.0_n100_lr0.001_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )

        file_base = ('SavedModels/base/'
                     'base_MNIST_lr0.001_augm_flagTrue_train_typeplain.pth'
                    )

        file_CEDA = ('SavedModels/base/'
                     'base_MNIST_lr0.001_augm_flagTrue_train_typeCEDA.pth'
                    )

        file_ODIN = ('SavedModels/odin/'
                     'base_MNIST_lr0.001_augm_flagTrue_train_typeplain_ODIN.pth'
                    )
        
        file_CCU_base = ('SavedModels/other/'
                         'ccu_base_MNIST.pth'
                        )
        
        file_CCU_ACET = ('SavedModels/'
                         'gmm__PCAMNIST_lam0.0_n100_lr0.001_lrgmm0.001_'
                         'augm_flagTrue_train_typeACET_GMMgrad_vars mu var.pth')
        
        file_CCU_double = ('SavedModels/'
                         'gmm__PCAMNIST_lam0.0_n100_lr0.001_lrgmm1e-05_'
                         'augm_flagTrue_train_typeCEDA_GMM_OUTgrad_vars mu var_OUT.pth')
        
        
        file_ACET = ('SavedModels/base/'
                     'base_MNIST_lr0.001_augm_flagTrue_train_typeACET_steps40.pth'
                    )
        
        file_GAN = ('SavedModels/other/gan/MNIST.pth')
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_MNIST.pth')
        
        file_maha = ('SavedModels/other/mahalanobis/MNIST.pth')
        
        file_MCDO = ('SavedModels/other/mcdo/MNIST.pth')
        
        
        
        files = [file_base, file_CEDA, file_ACET, 
                 file_GAN, file_MCDO, file_ODIN, 
                 file_maha, file_hendrycks, file_CCU_double, file_joint]
        
        keys = ['Base', 'CEDA', 'ACET', 'GAN', 'MCDO', 'ODIN', 'Maha', 'OE', 'CCUd', 'CCU']
        
        
        self.file_dict = collections.OrderedDict(zip(keys, files))

        
class FMNIST_models():
    def __init__(self):
        file_joint = ('SavedModels/'
                      'gmm__PCAFMNIST_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )

        file_base = ('SavedModels/base/'
                     'base_FMNIST_lr0.1_augm_flagTrue_train_typeplain.pth'
                    )

        file_CEDA = ('SavedModels/base/'
                     'base_FMNIST_lr0.1_augm_flagTrue_train_typeCEDA.pth'
                    )

        file_ODIN = ('SavedModels/odin/'
                     'base_FMNIST_lr0.1_augm_flagTrue_train_typeplain_ODIN.pth'
                    )
        
        file_CCU_base = ('SavedModels/other/'
                         'ccu_base_FMNIST.pth'
                        )
        
        file_CCU_ACET = ('SavedModels/'
                         'gmm__PCAFMNIST_lam0.0_n100_lr0.1_lrgmm1e-05'
                         '_augm_flagTrue_train_typeACET_GMMgrad_vars mu var.pth')
        
        file_ACET = ('SavedModels/base/'
                     'base_FMNIST_lr0.1_augm_flagTrue_train_typeACET_steps40.pth'
                    )
        
        file_GAN = ('SavedModels/other/gan/FMNIST.pth')
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_FMNIST.pth')
        
        file_maha = ('SavedModels/other/mahalanobis/FMNIST.pth')
        
        file_MCDO = ('SavedModels/other/mcdo/FMNIST.pth')
        
        
        files = [file_base, file_CEDA, file_ACET, 
                 file_GAN, file_MCDO, file_ODIN, 
                 file_maha, file_hendrycks, file_joint]
        
        keys = ['Base', 'CEDA', 'ACET', 'GAN', 'MCDO', 'ODIN', 'Maha', 'OE', 'CCU']
        
        self.file_dict = collections.OrderedDict(zip(keys, files))
       
    
class SVHN_models():
    def __init__(self):
        file_joint = ('SavedModels/'
                      'gmm__PCASVHN_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )

        file_base = ('SavedModels/base/'
                     'base_SVHN_lr0.1_augm_flagTrue_train_typeplain.pth'
                    )

        file_CEDA = ('SavedModels/base/'
                     'base_SVHN_lr0.1_augm_flagTrue_train_typeCEDA.pth'
                    )

        file_ODIN = ('SavedModels/odin/'
                     'base_SVHN_lr0.1_augm_flagTrue_train_typeplain_ODIN.pth'
                    )
        
        file_ACET = ('SavedModels/base/'
                     'base_SVHN_lr0.1_augm_flagTrue_train_typeACET_steps40.pth'
                    )
        
        file_CCU_base = ('SavedModels/other/'
                         'ccu_base_SVHN.pth'
                        )
        
        file_GAN = ('SavedModels/other/gan/SVHN.pth')
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_SVHN.pth')
        
        file_maha = ('SavedModels/other/mahalanobis/SVHN.pth')
        
        file_MCDO = ('SavedModels/other/mcdo/SVHN.pth')
        
        
        files = [file_base, file_CEDA, file_ACET, 
                 file_GAN, file_MCDO, file_ODIN, 
                 file_maha, file_hendrycks, file_joint]
        
        keys = ['Base', 'CEDA', 'ACET', 'GAN', 'MCDO', 'ODIN', 'Maha', 'OE', 'CCU']
        
        self.file_dict = collections.OrderedDict(zip(keys, files))


class CIFAR10_models():
    def __init__(self):
        file_joint = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )

        file_base = ('SavedModels/base/'
                     'base_CIFAR10_lr0.1_augm_flagTrue_train_typeplain.pth'
                    )

        file_CEDA = ('SavedModels/base/'
                     'base_CIFAR10_lr0.1_augm_flagTrue_train_typeCEDA.pth'
                    )

        file_ODIN = ('SavedModels/odin/'
                     'base_CIFAR10_lr0.1_augm_flagTrue_train_typeplain_ODIN.pth'
                    )
        
        file_ACET = ('SavedModels/base/'
                     'base_CIFAR10_lr0.1_augm_flagTrue_train_typeACET_steps40.pth'
                    )
        
        file_CCU_base = ('SavedModels/other/'
                         'ccu_base_CIFAR10.pth'
                        )
        
        file_CCU_double = ('SavedModels/'
                         'gmm__PCACIFAR10_lam0.0_n100_lr0.1_lrgmm1e-05_'
                         'augm_flagTrue_train_typeCEDA_GMM_OUTgrad_vars mu var_OUT.pth')
        
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_CIFAR10.pth')
        
        file_mahalanobis = ('SavedModels/other/mahalanobis/CIFAR10.pth')
        
        file_GAN = ('SavedModels/other/gan/CIFAR10.pth')
        
        file_hybrid = ('SavedModels/other/hybrid_CIFAR10.pth')
        
        file_maha = ('SavedModels/other/mahalanobis/CIFAR10.pth')
        
        file_MCDO = ('SavedModels/other/mcdo/CIFAR10.pth')
        
        
        
        files = [file_base, file_CEDA, file_ACET, 
                 file_GAN, file_MCDO, file_ODIN, 
                 file_maha, file_hendrycks, file_CCU_double, file_joint]
        
        keys = ['Base', 'CEDA', 'ACET', 'GAN', 'MCDO', 'ODIN', 'Maha', 'OE', 'CCUd', 'CCU']
        
        
        self.file_dict = collections.OrderedDict(zip(keys, files))


class CIFAR100_models():
    def __init__(self):
        file_joint = ('SavedModels/'
                      'gmm__PCACIFAR100_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )

        file_base = ('SavedModels/base/'
                     'base_CIFAR100_lr0.1_augm_flagTrue_train_typeplain.pth'
                    )

        file_CEDA = ('SavedModels/base/'
                     'base_CIFAR100_lr0.1_augm_flagTrue_train_typeCEDA.pth'
                    )

        file_ODIN = ('SavedModels/odin/'
                     'base_CIFAR100_lr0.1_augm_flagTrue_train_typeplain_ODIN.pth'
                    )
        
        file_ACET = ('SavedModels/base/'
                     'base_CIFAR100_lr0.1_augm_flagTrue_train_typeACET_steps40.pth'
                    )
        
        file_CCU_base = ('SavedModels/other/'
                         'ccu_base_CIFAR100.pth'
                        )
        
        file_GAN = ('SavedModels/other/gan/CIFAR100.pth')
        
        file_hendrycks = ('SavedModels/other/outlier-exposure/hendrycks_CIFAR100.pth')
        
        file_maha = ('SavedModels/other/mahalanobis/CIFAR100.pth')
        
        
        file_MCDO = ('SavedModels/other/mcdo/CIFAR100.pth')
        
        
        files = [file_base, file_CEDA, file_ACET, 
                 file_GAN, file_MCDO, file_ODIN, 
                 file_maha, file_hendrycks, file_joint]
        
        keys = ['Base', 'CEDA', 'ACET', 'GAN', 'MCDO', 'ODIN', 'Maha', 'OE', 'CCU']
        
        self.file_dict = collections.OrderedDict(zip(keys, files))
        

model_dict =  {'MNIST':          MNIST_models,
               'FMNIST':         FMNIST_models,
               'SVHN':           SVHN_models,
               'CIFAR10':        CIFAR10_models,
               'CIFAR100':       CIFAR100_models,
              }