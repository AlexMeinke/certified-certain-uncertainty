import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score
import utils.dataloaders as dl


class LeNetTemp(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.*temperature), 
                                        requires_grad=False)
        self.model = model
        
    def forward(self, x):
        x = F.relu(self.model.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.model.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.model.fc1(x))
        x = self.model.fc2(x)
        x = F.log_softmax(x / self.temperature, dim=1)
        return x

    
class ResNetTemp(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.*temperature), 
                                        requires_grad=False)
        self.model = model
        
    def forward(self, x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.model.linear(out)
        return F.log_softmax(out / self.temperature, dim=1)

    
class ModelODIN(nn.Module):
    def __init__(self, model, epsilon, device=torch.device('cpu')):
        super().__init__()
        self.epsilon = epsilon
        self.device = device
        self.model = model.to(device)
        
    def forward(self, x):
        x = self.FGSM(x)
        x = self.model(x)
        return x
    
    def FGSM(self, x):
        with torch.enable_grad():
            x = x.requires_grad_()
            y = self.model(x)
            losses = y.max(1)[0]
            loss = losses.sum()
            
            grad = torch.autograd.grad (losses.sum(), x)[0]
  
        x = x + self.epsilon * grad
        x = torch.clamp(x, 0, 1).requires_grad_()
        return x


def grid_search_variables(base_model, model_params, device, out_seeds=False):
    shape = next(iter(model_params.cali_loader))[0][0].shape
    temperatures = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    #temperatures = [200, 500, 1000]

    epsilons = np.linspace(0, 0.004, 21)

    grid = []

    for T in temperatures:
        vec = []
        temp_model = model_dict[model_params.data_name](base_model, T)
        for eps in epsilons:
            model = ModelODIN(temp_model, eps, device)
            stats = aggregate_stats([model], device, shape, 
                                    classes=model_params.classes,
                                    out_seeds=out_seeds)
            auroc, fp95 = get_auroc([model], model_params, stats, device)
            vec.append(auroc + fp95)
        grid.append(vec)
        
    auroc = torch.tensor(grid)[:,:,0]
    fp95 = torch.tensor(grid)[:,:,1]

    ind = auroc.view(-1).argmax().item()

    xv, yv = np.meshgrid(epsilons, temperatures)
    T = yv.reshape(-1)[ind]
    eps = xv.reshape(-1)[ind]
    
    odin_model = ModelODIN(model_dict[model_params.data_name](base_model, T), 
                                               eps, device=device)
    stats = aggregate_stats([model], device, shape, 
                            classes=model_params.classes)
    
    return odin_model, T, eps


def get_auroc(model_list, model_params, stats, device):
    auroc = []
    fp95 = []
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

        auroc.append(roc_auc_score(y_true, y_scores))
        fp95.append( ((stats[i] > 0.95).float().mean()).item() )
    return auroc, fp95


def aggregate_stats(model_list, device, shape, classes=10, 
                    batches=20, batch_size=100, out_seeds=False):
    stats = []
    
    if out_seeds:
        if shape[0]==1:
            dataset = 'MNIST'
        else:
            dataset = 'CIFAR10'
        out_loader = iter(dl.TinyImages(dataset, batch_size=batch_size))

    for _ in range(batches):
        seed = torch.rand((batch_size,)+tuple(shape), device=device)

        if out_seeds:
            seed = next(out_loader)[0].to(device)
        else:
            seed = torch.rand((batch_size,) + tuple(shape), device=device)
            
        batch_stats = []
        for i, model in enumerate(model_list):
            model.eval()
            batch_stats.append(model(seed).max(1)[0].exp().detach().cpu().clone())
            
        batch_stats = torch.stack(batch_stats, 0)
        stats.append(batch_stats.clone())

    stats = torch.cat(stats, -1)
    
    return stats


model_dict = { 'MNIST':          LeNetTemp,
               'FMNIST':         ResNetTemp,
               'SVHN':           ResNetTemp,
               'CIFAR10':        ResNetTemp,
               'CIFAR100':       ResNetTemp,
              }