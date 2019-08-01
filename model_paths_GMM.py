import collections


class MNIST_models():
    def __init__(self):
        file_1 = ('SavedModels/'
                      'gmm__PCAMNIST_lam0.0_n10_lr0.001_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_10 = ('SavedModels/'
                      'gmm__PCAMNIST_lam0.0_n10_lr0.001_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_100 = ('SavedModels/'
                      'gmm__PCAMNIST_lam0.0_n100_lr0.001_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_1000 = ('SavedModels/'
                      'gmm__PCAMNIST_lam0.0_n1000_lr0.001_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )

        
        
        files = [file_1, file_10, file_100, file_1000]
        keys = ['1', '10', '100', '1000']
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
        
        files = [file_base, file_CEDA, file_ACET, file_ODIN, file_CCU_base, file_joint, file_CCU_ACET]
        keys = ['Base', 'CEDA', 'ACET', 'ODIN', 'CCUb', 'CCU', 'CCUA']
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
        
        files = [file_base, file_CEDA, file_ACET, file_ODIN, file_CCU_base, file_joint]
        keys = ['Base', 'CEDA', 'ACET', 'ODIN', 'CCUb', 'CCU']
        self.file_dict = collections.OrderedDict(zip(keys, files))


class CIFAR10_models():
    def __init__(self):
        file_1 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n1_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_5 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n5_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_10 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n10_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_20 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n20_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_30 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n30_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_50 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n50_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_100 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n100_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_200 = ('SavedModels/'
                      'gmm__PCACIFAR10_lam0.0_n200_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        file_500 = ('Checkpoints/'
                      'gmm__PCACIFAR10_lam0.0_n500_lr0.1_lrgmm1e-05_'
                      'augm_flagTrue_train_typeCEDA_GMMgrad_vars mu var.pth'
                     )
        
        
        files = [file_1, file_5, file_10, file_20, file_30, file_50, file_100, file_200, file_500]
        keys = ['1','5','10','20','30','50','100','200','500']

        
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
        
        files = [file_base, file_CEDA, file_ACET, file_ODIN, file_CCU_base, file_joint]
        keys = ['Base', 'CEDA', 'ACET', 'ODIN', 'CCUb', 'CCU']
        self.file_dict = collections.OrderedDict(zip(keys, files))
        

model_dict =  {'MNIST':          MNIST_models,
               'FMNIST':         FMNIST_models,
               'SVHN':           SVHN_models,
               'CIFAR10':        CIFAR10_models,
               'CIFAR100':       CIFAR100_models,
              }