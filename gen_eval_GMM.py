import torch
import model_paths_GMM as model_paths
import model_params as params
import utils.eval as ev
import pandas as pd
import utils.traintest as tt
import datetime
import argparse


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--dataset', type=str, required=True, help='Which dataset to use.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

hps = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:' + str(hps.gpu))
    
model_params = params.params_dict[hps.dataset]()
model_path = model_paths.model_dict[hps.dataset]() 
model_list = [torch.load(file).to(device) for file in model_path.file_dict.values()]

accuracies = [tt.test(model, device, model_params.test_loader, min_conf=.001)[0]
              for model in model_list]
results = [ev.evaluate(model, device, model_params.data_name, model_params.loaders) 
           for model in model_list]
test_error = [100*(1.-acc) for acc in accuracies]


keys = [key + ' ({:.2f}%)'.format(te) for (te, key) in zip(test_error, model_path.file_dict.keys())]
df = pd.concat(results, axis=1, keys=keys)

time = str(datetime.datetime.now())
df.to_csv('results/gmm_' + hps.dataset + time + '.csv')
df.to_pickle('results/gmm_' + hps.dataset + time)

file = open('results/gmm_' + hps.dataset + time + '.txt','w') 
file.write(df.round(1).to_latex())
file.close() 
