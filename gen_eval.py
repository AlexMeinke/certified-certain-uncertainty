'''
    Generates tables to evaluate out-of-distribution detection performance of models
    specified in model_paths.py using datasets specified in model_params.py
    Output goes to results/ as .csv, .txt (latex code) and as pickle (for pandas to recover it)
'''

import torch
import model_paths
import model_params as params
import utils.eval as ev
import pandas as pd
import utils.traintest as tt

import datetime
import argparse


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--dataset', type=str, required=True, help='Which dataset to use.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--drop_mmc', type=bool, default=False, 
                    help='whether to use the more compact format.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--aupr', type=bool, default=False, 
                    help='If True, then computes AUPR, else uses AUROC.')

hps = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:' + str(hps.gpu))
    
model_params = params.params_dict[hps.dataset](batch_size=hps.batch_size)
model_path = model_paths.model_dict[hps.dataset]() 
model_list = [torch.load(file).to(device) for file in model_path.file_dict.values()]

accuracies = [tt.test(model, device, model_params.test_loader, min_conf=.0)[0]
              for model in model_list]

results = [ev.evaluate(model, device, model_params.data_name, 
                       model_params.loaders, drop_mmc=hps.drop_mmc, aupr=hps.aupr) 
           for model in model_list]

test_error = [100*(1.-acc) for acc in accuracies]

if hps.drop_mmc:
    keys = [key for (te, key) in zip(test_error, model_path.file_dict.keys())]
else:
    keys = [key + ' ({:.2f}%)'.format(te) for (te, key) in zip(test_error, model_path.file_dict.keys())]
df = pd.concat(results, axis=1, keys=keys)

time = str(datetime.datetime.now())

file = 'results/' + hps.dataset + '_'
if hps.aupr:
    file += 'aupr_'
file += time

df.to_csv(file + '.csv')
df.to_pickle(file)

file = open(file + '.txt','w') 
file.write(df.round(1).to_latex())
file.close() 
