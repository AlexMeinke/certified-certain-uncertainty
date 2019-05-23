import torch
import pandas as pd

from sklearn.metrics import roc_auc_score

import model_paths
import model_params as params
import utils.eval as ev

import datetime
import argparse


parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=3, help='GPU index.')
parser.add_argument('--steps', type=int, default=200, help='num of attack steps.')
parser.add_argument('--restarts', type=int, default=10, help='num of restarts in attack.')
parser.add_argument('--alpha', type=float, default=3., help='initial step size.')
parser.add_argument('--batches', type=int, default=10, help='number of batches to test on.')
parser.add_argument('--batch_size', type=int, default=100, help='size of batches that one tests on.')
parser.add_argument('--datasets', nargs='+', type=str, required=True, 
                    help='datasets to run attack on.')


hps = parser.parse_args()

datasets = hps.datasets
steps = hps.steps
alpha = hps.alpha
restarts = hps.restarts
batches = hps.batches
batch_size = hps.batch_size

device = torch.device('cuda:' + str(hps.gpu))

saving_string = ('samples_steps' + str(steps) 
                 + '_alpha' + str(alpha) 
                 + '_restarts' + str(restarts)
                 + '_batches' + str(batches)
                 + '_batch_size' + str(batch_size)
                )
for dataset in datasets:
    saving_string += '_' + dataset
    
saving_string += '_' + str(datetime.datetime.now())

def get_auroc(model_list, model_params, stats, device):
    auroc = []
    success_rate = []
    conf_list = []
    for i, model in enumerate(model_list):
        with torch.no_grad():
            conf = []
            for data, _ in model_params.test_loader:
                data = data.to(device)

                output = model(data).max(1)[0].exp()

                conf.append(output.cpu())

        conf = torch.cat(conf, 0)

        y_true = torch.cat([torch.ones_like(conf.cpu()), 
                            torch.zeros_like(stats[i])]).cpu().numpy()
        y_scores = torch.cat([conf.cpu(), 
                              stats[i]]).cpu().numpy()
        success_rate.append((stats[i] > conf.mean()).float().mean().item())
        auroc.append(roc_auc_score(y_true, y_scores))
        conf_list.append(conf)
    return auroc, success_rate, conf_list


auroc_vec = []
mmc_vec = []
success_rate_vec = []

for dataset in datasets:
    model_params = params.params_dict[dataset]()
    model_path = model_paths.model_dict[dataset]() 
    model_list = [torch.load(file).to(device) for file in model_path.files]
    
    gmm = model_list[-1].mm

    shape = enumerate(model_params.cali_loader).__next__()[1][0][0].shape
    
    stats, bounds, seeds, samples = ev.aggregate_adv_stats(model_list, gmm, device, 
                                           shape, classes=model_params.classes, 
                                           batches=batches, batch_size=batch_size, 
                                           steps=steps, 
                                           restarts=restarts, alpha=alpha)
    cont  = ev.StatsContainer(stats, bounds, seeds, samples)
    torch.save(cont, 'results/backup/' + saving_string + '_' + dataset + '.pth')
    auroc, success_rate, conf = get_auroc(model_list, model_params, stats, device)
    
    auroc_vec.append(auroc)
    success_rate_vec.append(success_rate)
    mmc_vec.append([stats[i].mean() for i in range(len(model_list))])
    
stats = torch.stack([torch.tensor(mmc_vec), 
                     torch.tensor(success_rate_vec),
                     torch.tensor(auroc_vec)], 2).transpose(0,1)

df_list = []

for i in range(len(model_list)):
    df = pd.DataFrame(stats[i].numpy() )
    df.insert(0, 'A', pd.Series(datasets))
    df.columns = ['DataSet', 'MMC', 'SR', 'AUROC']
    df_list.append(df.set_index('DataSet'))
    
df = pd.concat(df_list, axis=1, keys=model_path.keys)

df.to_csv('results/' + saving_string + '.csv')

file = open('results/' + saving_string + '.txt','w') 
file.write(df.round(3).to_latex())
file.close() 