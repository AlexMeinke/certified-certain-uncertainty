import torch
import pandas as pd

from sklearn.metrics import roc_auc_score

import model_paths
import model_params as params
import utils.eval as ev
import utils.traintest as tt

import datetime
import argparse

import collections

parser = argparse.ArgumentParser(description='Define hyperparameters.', prefix_chars='-')

parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--steps', type=int, default=500, help='num of attack steps.')
parser.add_argument('--restarts', type=int, default=50, help='num of restarts in attack.')
parser.add_argument('--alpha', type=float, default=3., help='initial step size.')
parser.add_argument('--batches', type=int, default=2, help='number of batches to test on.')
parser.add_argument('--batch_size', type=int, default=100, help='size of batches that one tests on.')
parser.add_argument('--datasets', nargs='+', type=str, required=True, 
                    help='datasets to run attack on.')
parser.add_argument('--drop_mmc', type=bool, default=False, 
                    help='when active, only displays AUC and Success rate.')
parser.add_argument('--vertical', type=bool, default=False, 
                    help='when active, aranges models vertically.')
parser.add_argument('--wide_format', type=bool, default=False, 
                    help='when active, aranges MMC, AUC, SR, TE in separate rows.')
parser.add_argument('--fit_out', type=bool, default=False, 
                    help='use GMM for the out-distribution as well.')
parser.add_argument('--out_seeds', type=bool, default=False, 
                    help='use 80m tiny images as seeds for attack.')

hps = parser.parse_args()


datasets = hps.datasets
steps = hps.steps
alpha = hps.alpha
restarts = hps.restarts
batches = hps.batches
batch_size = hps.batch_size


if torch.cuda.is_available():
    device = torch.device('cuda:' + str(hps.gpu))

saving_string = ('samples_steps' + str(steps) 
                 + '_alpha' + str(alpha) 
                 + '_restarts' + str(restarts)
                 + '_batches' + str(batches)
                 + '_batch_size' + str(batch_size)
                )

for dataset in datasets:
    saving_string += '_' + dataset
    
if hps.out_seeds:
    saving_string += '_OUTSEEDS'
    
saving_string += '_' + str(datetime.datetime.now())


def get_auroc(model_list, model_params, stats, device):
    auroc = []
    success_rate = []
    conf_list = []
    for i, model in enumerate(model_list):
        #print(i)
        with torch.no_grad():
            conf = []
            for data, _ in model_params.test_loader:
                data = data.to(device)

                output = model(data).max(1)[0]

                conf.append(output.cpu())

        conf = torch.cat(conf, 0)

        y_true = torch.cat([torch.ones_like(conf.cpu()), 
                            torch.zeros_like(stats[i])]).cpu().numpy()
        y_scores = torch.cat([conf.cpu(), 
                              stats[i]]).cpu().numpy()
        success_rate.append((stats[i] >= conf.median()).float().mean().item())
        auroc.append(roc_auc_score(y_true, y_scores))
        conf_list.append(conf.exp())
    return auroc, success_rate, conf_list


acc_vec = []
auroc_vec = []
mmc_vec = []
success_rate_vec = []


for dataset in datasets:
    model_params = params.params_dict[dataset]()
    model_path = model_paths.model_dict[dataset]() 
    model_list = [torch.load(file).to(device) for file in model_path.file_dict.values()]
    
    if hps.fit_out:
        model_list = model_list[:-1]
    
    accuracies = [tt.test(model, device, model_params.test_loader, min_conf=.001)[0]
                  for model in model_list]
       
    shape = next(iter(model_params.cali_loader))[0][0].shape
    
    if hps.fit_out:
        gmm = model_list[-1].mm
        gmm_out = model_list[-1].mm_out
        

        results = ev.aggregate_adv_stats_out(model_list, gmm, gmm_out, device, 
                                             shape, classes=model_params.classes, 
                                             batches=batches, batch_size=batch_size, 
                                             steps=steps, out_seeds=hps.out_seeds,
                                             restarts=restarts, alpha=alpha)
        stats, bounds, seeds, samples = results
        
    else:
        gmm = model_list[-1].mm

        stats, bounds, seeds, samples = ev.aggregate_adv_stats(model_list, gmm, device, 
                                               shape, classes=model_params.classes, 
                                               batches=batches, batch_size=batch_size, 
                                               steps=steps, 
                                               restarts=restarts, alpha=alpha)
    cont  = ev.StatsContainer(stats, bounds, seeds, samples)
    torch.save(cont, 'results/backup/' + saving_string + '_' + dataset + '.pth')
    
    auroc, success_rate, conf = get_auroc(model_list, model_params, stats, device)
    
    acc_vec.append(accuracies)
    auroc_vec.append(auroc)
    success_rate_vec.append(success_rate)
    mmc_vec.append([stats[i].exp().mean() for i in range(len(model_list))])
    
    
if hps.drop_mmc:
    stats = 100 * torch.stack([
                               torch.tensor(success_rate_vec),
                               torch.tensor(auroc_vec)], 2).transpose(0,1)
else:
    stats = 100 * torch.stack([torch.tensor(mmc_vec), 
                               torch.tensor(success_rate_vec),
                               torch.tensor(auroc_vec)], 2).transpose(0,1)

    
if hps.wide_format:
    print(acc_vec)
    stats = 100 * torch.stack([1.-torch.tensor(acc_vec),
                               torch.tensor(mmc_vec), 
                               torch.tensor(success_rate_vec),
                               torch.tensor(auroc_vec)], 2).transpose(0,1)
    
    if hps.fit_out:
        stats_dict = collections.OrderedDict(zip(list(model_path.file_dict.keys())[:-1], 
                                             stats[:,0,:].tolist()))
    else:
        stats_dict = collections.OrderedDict(zip(model_path.file_dict.keys(), 
                                                 stats[:,0,:].tolist()))
    
    metrics = ['TE', 'MMC', 'SR', 'AUC']
    df = pd.DataFrame(stats_dict, index=[len(metrics)*[hps.datasets[0]], metrics])
else:
    df_list = []

    for i in range(len(model_list)):
        df = pd.DataFrame(stats[i].numpy() )
        df.insert(0, 'A', pd.Series(datasets))
        if hps.drop_mmc:
            df.columns = ['DataSet', 'SR', 'AUC']
        else:
            df.columns = ['DataSet', 'MMC', 'SR', 'AUC']

        df_list.append(df.set_index('DataSet'))

    if hps.fit_out:
        df = pd.concat(df_list, axis=1, keys=list(model_path.file_dict.keys())[:-1])
    else:
        if hps.vertical:
            df = pd.concat(df_list, axis=0, keys=list(model_path.file_dict.keys()))
        else:
            df = pd.concat(df_list, axis=1, keys=list(model_path.file_dict.keys()))

        
df.to_csv('results/' + saving_string + '.csv')

file = open('results/' + saving_string + '.txt','w') 
file.write(df.round(1).to_latex())
file.close()
