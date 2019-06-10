import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

import utils.models as models
import utils.plotting as plotting
import utils.adversarial as adv
import utils.eval as ev
import utils.gmm_helpers as gmm_helpers 
import model_params as params
import utils.resnet_orig as resnet
import model_paths

import datetime
import argparse


parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--steps', type=int, default=200, help='num of attack steps.')
parser.add_argument('--restarts', type=int, default=10, help='num of restarts in attack.')
parser.add_argument('--alpha', type=float, default=3., help='initial step size.')
parser.add_argument('--datasets', nargs='+', type=str, required=True, 
                    help='datasets to run attack on.')

hps = parser.parse_args()

steps = hps.steps
alpha = hps.alpha
restarts = hps.restarts
batch_size = 2

saving_string = ('samples_steps' + str(steps) 
                 + '_alpha' + str(alpha) 
                 + '_restarts' + str(restarts)
                 + '_' + str(datetime.datetime.now())
                )

datasets = hps.datasets
device = torch.device('cuda:' + str(hps.gpu))

for ds in datasets:
    saving_string += '_' + ds


plt.figure(figsize=(10,10))

plot_rows = len(datasets)

for j_row, dataset in enumerate(datasets):
    model_params = params.params_dict[dataset]()
    model_path = model_paths.model_dict[dataset]() 
    model_list = [torch.load(file).to(device) for file in model_path.file_dict.values()]
    gmm = model_list[-1].mm


    shape = enumerate(model_params.cali_loader).__next__()[1][0][0].shape

    pca = models.MyPCA(gmm.metric.comp_vecs.t(), gmm.metric.singular_values, shape)
    
    seed = torch.rand((batch_size,) + shape, device=device)

    lam = 1.
    f = 1.1
    b = lam * (f-1.) / (model_params.classes-f)

    batch_bounds = []

    for x in seed:
        batch_bounds.append( scipy.optimize.brentq(gmm_helpers.get_b, 0, 10000., args = (x, gmm, b)) )
    batch_bounds = torch.tensor(batch_bounds, device=device)
    
    noise_list = []

    for model in model_list:
        adv_noise, loss = adv.gen_pca_noise(model, device, seed, pca, batch_bounds, 
                                            restarts=restarts, perturb=True, 
                                            steps=steps, alpha=alpha)
        noise_list.append(adv_noise)
        
    Y = [model(noise).max(1) for (noise, model) in zip(noise_list, model_list)]
    conf = [y[0][0].exp().item() for y in Y]
    pred = [y[1][0].item() for y in Y]

    noise = [noise[0] for noise in noise_list]
    
    n = len(pred)
    n_plots = len(pred)+1
    init = 2
    
    plt.subplot(plot_rows, n_plots, 1 + n_plots*j_row)
    plt.ylabel(dataset, rotation=90, horizontalalignment='left', va='center')
    
    if dataset in ['MNIST', 'FMNIST']:
        plt.imshow(seed[0].squeeze().detach().cpu(), cmap='gray', interpolation='none')
    elif dataset in ['CIFAR10', 'SVHN', 'CIFAR100']:
        plt.imshow(seed[0].transpose(0,2).transpose(0,1).detach().cpu(), interpolation='none')

    plt.xticks([])
    plt.yticks([])
        
    for i in range(n):
        plt.subplot(plot_rows, n_plots, init+ n_plots*j_row + i)
        
        if dataset=='MNIST':
            string = list(model_path.file_dict.keys())[i] + '\n'
        else:
            string = ''
        
        string += ( str(plotting.classes_dict[dataset][pred[i]] )
                   + ": %.3f" % conf[i] )

        #plt.title(string, fontsize='small')
        plt.title(string)
        
        if dataset in ['MNIST', 'FMNIST']:
            plt.imshow(noise[i].squeeze().detach().cpu(), cmap='gray', interpolation='none')
        elif dataset in ['CIFAR10', 'SVHN', 'CIFAR100']:
            plt.imshow(noise[i].transpose(0,2).transpose(0,1).detach().cpu(), interpolation='none')
        plt.xticks([])
        plt.yticks([])

#plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=.1, hspace=.1)
plt.tight_layout()
myplot = plt.gcf()

myplot.savefig('results/' + saving_string + '.eps', format='eps')
