import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    
class DeepEnsemble(nn.Module):
    def __init__(self, model_list):
        super().__init__()
        self.M = len(model_list)
        self.norm_const = np.log(self.M)
        
        self.models = ListModule(*model_list)
        
    def forward(self, x):
        out = []
        
        for model in self.models:
            out.append(model(x))
            
        out = torch.stack(out, 0)
        out = torch.logsumexp(out, 0) - self.norm_const
        return out