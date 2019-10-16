'''
        Monte-Carlo Dropout (MCD)
        
    Code implementing https://arxiv.org/abs/1506.02142 for classification
    Uses VGG because ResNets originally don't have Dropout layers
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


class MC_dropout(nn.Dropout):
    def __init__(self, p=0.5, inplace=False):
        super().__init__(p, inplace)
        
    def eval(self):
        return
    
    
class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            MC_dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            MC_dropout(),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def logit(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

                
def make_layers(cfg, in_channels):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg13(pretrained=False, in_channels=3, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], in_channels=in_channels), **kwargs)
    return model


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = MC_dropout()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def logit(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

class MC_Model(nn.Module):
    def __init__(self, model, iterations=7, classes=10):
        super().__init__()
        self.model = model
        self.iterations = iterations
        
        self.class_vec = nn.Parameter(torch.arange(classes)[None,:], requires_grad=False)

    def forward(self, x):
        out = []
        for _ in range(self.iterations):
            out.append(self.model(x).exp())
        out = torch.stack(out)
        y = out.mean(0)
        uncertainty = ((out - y[None,:,:])**2).mean(0).sum(1)
        
        idx = y.max(1)[1]
        idx = (self.class_vec==idx[:,None])
        out = -np.inf * torch.ones_like(y)
        out[idx] = -uncertainty
        return out
    

# I tested this as well to make sure that the bad OOD performance didn't come from
# using the softmax output. This actually worked worse
class MC_Model_logit(nn.Module):
    def __init__(self, model, iterations=7, classes=10):
        super().__init__()
        self.model = model
        self.iterations = iterations
        
        self.class_vec = nn.Parameter(torch.arange(classes)[None,:], requires_grad=False)

    def forward(self, x):
        out = []
        for _ in range(self.iterations):
            out.append(self.model.logit(x))
        out = torch.stack(out)
        y = out.mean(0)
        uncertainty = ((out - y[None,:,:])**2).mean(0).mean(1)
        
        idx = y.max(1)[1]
        idx = (self.class_vec==idx[:,None])
        out = -1000. * torch.ones_like(y)
        out[idx] = -uncertainty
        return out
    