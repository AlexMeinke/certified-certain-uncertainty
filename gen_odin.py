import torch
import model_params as params
import utils.odin as odin

import argparse


parser = argparse.ArgumentParser(description='Parse arguments.', prefix_chars='-')

parser.add_argument('--dataset', type=str, required=True, help='Which dataset to use.')
parser.add_argument('--path', type=str, required=True, help='Path of the plain model.')
parser.add_argument('--gpu', type=int, default=3, help='GPU index.')

hps = parser.parse_args()

device = torch.device('cuda:' + str(hps.gpu))

model_params = params.params_dict[hps.dataset]()

base_model = torch.load('SavedModels/base/' + hps.path + '.pth').to(device)
ODIN_model, _, _ = odin.grid_search_variables(base_model, model_params, device)

torch.save(ODIN_model.cpu(), 'SavedModels/odin/' + hps.path + '_ODIN.pth')
