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
        
        self.files = [file_base, file_CEDA, file_ODIN, file_joint]
        self.keys = ['Base', 'CEDA', 'ODIN', 'CCU']

        
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
        
        self.files = [file_base, file_CEDA, file_ODIN, file_joint]
        self.keys = ['Base', 'CEDA', 'ODIN', 'CCU']
       
    
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
        
        self.files = [file_base, file_CEDA, file_ODIN, file_joint]
        self.keys = ['Base', 'CEDA', 'ODIN', 'CCU']


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
        
        self.files = [file_base, file_CEDA, file_ODIN, file_joint]
        self.keys = ['Base', 'CEDA', 'ODIN', 'CCU']


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
        
        self.files = [file_base, file_CEDA, file_ODIN, file_joint]
        self.keys = ['Base', 'CEDA', 'ODIN', 'CCU']


model_dict =  {'MNIST':          MNIST_models,
               'FMNIST':         FMNIST_models,
               'SVHN':           SVHN_models,
               'CIFAR10':        CIFAR10_models,
               'CIFAR100':       CIFAR100_models,
              }