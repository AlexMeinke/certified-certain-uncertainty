import torch
import model_params as params
import utils.odin as odin
import model_paths

import argparse


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--dataset', type=str, required=True, help='Which dataset to use.')
parser.add_argument('--path', type=str, default=None, help='Path of the plain model.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--out_seeds', type=bool, default=0, help='whether to calibrate on 80Mtinyimages.')

hps = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:' + str(hps.gpu))
    
if hps.path is None:
    hps.path = model_paths.model_dict[hps.dataset]().file_dict['Base']

model_params = params.params_dict[hps.dataset]()

base_model = torch.load(hps.path).to(device)
ODIN_model, _, _ = odin.grid_search_variables(base_model, model_params, 
                                              device, out_seeds=hps.out_seeds)

saving_string = hps.dataset
if hps.out_seeds:
    saving_string += '_OUTSEEDS'
torch.save(ODIN_model.cpu(), 'SavedModels/odin/' + saving_string + '_ODIN.pth')
