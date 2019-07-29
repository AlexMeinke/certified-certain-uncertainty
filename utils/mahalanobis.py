'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score

from torch.autograd import Variable
from torch.nn.parameter import Parameter

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y
    
    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
       # return y, penultimate.view(y.shape[0], -1)
        return y, y
    
def ResNet18(num_c):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_c)

def ResNet34(num_c):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())


class ModelODIN(nn.Module):
    def __init__(self, model, epsilon, mu, var, device=torch.device('cpu')):
        super().__init__()
        self.epsilon = epsilon
        self.device = device
        self.model = model.to(device)
        self.mu = nn.Parameter(mu.to(device), requires_grad=False)
        self.invsig = nn.Parameter(var.inverse().to(device), requires_grad=False)
        
    def forward(self, x):
        x = self.FGSM(x)
     #   x = self.model(x)
        
        _, out = self.model.penultimate_forward(x)
        diff = out[:,None,:] - self.mu[None,:,:]
        y = - (diff * (self.invsig[None,None,:,:] * diff[:,:,None,:]).sum(-1)).sum(-1)
        
        return y
    
    def FGSM(self, x):
        with torch.enable_grad():
            x = x.requires_grad_()
            _, out = self.model.penultimate_forward(x)
            diff = out[:,None,:] - self.mu[None,:,:]
            y = - (diff * (self.invsig[None,None,:,:] * diff[:,:,None,:]).sum(-1)).sum(-1)
            losses = y.max(1)[0]
            loss = losses.sum()
            
            grad = torch.autograd.grad (losses.sum(), x)[0]
  
        x = x + self.epsilon * grad
        x = torch.clamp(x, 0, 1).requires_grad_()
        return x

    
def compute_layer_statistics(model, device, loader):
    model.eval()
    classes = 10
    class_num = ((torch.tensor(loader.dataset.targets)[:,None]
                  == torch.arange(10)[None,:]).sum(0).float())
    
    with torch.no_grad():
    
        data = enumerate(loader).__next__()[1][0].to(device)
        _, act = model.penultimate_forward(data)
        act = [act]

        dims = [a[0].view(-1).shape[0] for a in act]
        mean = [torch.zeros(classes, d) for d in dims]
        cov = [torch.zeros(d, d) for d in dims]

        for data, label in loader:
            data, label = data.to(device), label.to(device)
            label = torch.zeros(label.shape[0], classes, 
                                device=device).scatter_(1, label[:,None], 1)

            _, act = model.penultimate_forward(data)

            for i, layer_act in enumerate([act]):
                mean[i] += (layer_act.view(layer_act.shape[0], -1)[:,None,:]
                            * label[:,:,None]).sum(0).cpu()
        for i, _ in enumerate(mean):
            mean[i] = mean[i] / class_num[:,None]
        
        for idx, (data, label) in enumerate(loader):
            print(idx)
            data = data.to(device)
            label = label.to(device)
            
            _, act = model.penultimate_forward(data)

            for i, layer_act in enumerate([act]):
                diff = (layer_act.view(layer_act.shape[0], -1).cpu() - mean[i][label])
                cov[i] += (diff[:,:,None] * diff[:,None,:]).sum(0)
                
        for i, _ in enumerate(cov):
            cov[i] = cov[i] / class_num.sum()
        return mean, cov


def grid_search_variables(base_model, model_params, device):
    mean, cov = compute_layer_statistics(base_model, device, model_params.train_loader)
    
    shape = enumerate(model_params.cali_loader).__next__()[1][0][0].shape
    temperatures = [1]
    epsilons = np.linspace(0, 0.004, 21)

    grid = []

    for T in temperatures:
        vec = []
        for eps in epsilons:
            model = ModelODIN(base_model, eps, mean[0], cov[0], device)
            stats = aggregate_stats([model], device, shape, 
                                    classes=model_params.classes)
            auroc, fp95 = get_auroc([model], model_params, stats, device)
            vec.append(auroc + fp95)
        grid.append(vec)
        
    auroc = torch.tensor(grid)[:,:,0]
    fp95 = torch.tensor(grid)[:,:,1]

    ind = auroc.view(-1).argmax().item()

    xv, yv = np.meshgrid(epsilons, temperatures)
    T = yv.reshape(-1)[ind]
    eps = xv.reshape(-1)[ind]
    
    odin_model = ModelODIN(base_model, eps, mean[0], cov[0], device)
    
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
                    batches=20, batch_size=100):
    stats = []

    for _ in range(batches):
        seed = torch.rand((batch_size,)+tuple(shape), device=device)
        
        batch_stats = []
        for i, model in enumerate(model_list):
            model.eval()
            batch_stats.append(model(seed).max(1)[0].exp().detach().cpu().clone())
            
        batch_stats = torch.stack(batch_stats, 0)
        stats.append(batch_stats.clone())

    stats = torch.cat(stats, -1)
    
    return stats
