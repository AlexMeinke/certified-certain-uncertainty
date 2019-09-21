from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, cdist, squareform
import utils.dataloaders as dl

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


class ModelODIN(nn.Module):
    def __init__(self, model, params, device=torch.device('cpu')):
        super().__init__()
        self.epsilon = 0
        self.device = device
        self.model = model.to(device)
        
        self.grid_search_variables(params, device)
        
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
    
    def grid_search_variables(self, model_params, device):
        #shape = next(iter(model_params.train_loader))[0][0].shape

        #epsilons = np.linspace(0, 0.004, 5)
        epsilons = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]

        vec = []
        alpha_vec = []
        for eps in epsilons:
            self.epsilon  = eps
            
            _, auroc, fp95 = test_metrics(self, device, 
                                          model_params.train_loader,
                                          model_params.tinyimage_loader)
            
            vec.append([auroc, fp95])
        
        auroc = torch.tensor(vec)[:,0]
        fp95 = torch.tensor(vec)[:,1]
        
        
        ind = auroc.view(-1).argmax().item()

        self.epsilon  = epsilons[ind]


### Mahalanobis Detector

class Mahalanobis(nn.Module):
    def __init__(self, model, model_params, device=torch.device('cpu')):
        super().__init__()
        self.model = model.to(device)
        
        self.device = device
        
        data = next(iter(model_params.train_loader))[0].to(device)
        dim = self.model.penultimate(data)[1].shape[1]
        
        means, prec = self.sample_estimator(model_params.classes, [dim],
                                           model_params.train_loader, device)
        
        self.means = nn.Parameter(means[0].to(device), requires_grad=False)
        self.precision = nn.Parameter(prec[0].to(device), requires_grad=False)
        
        self.class_vec = nn.Parameter(torch.arange(model_params.classes)[None,:], 
                                      requires_grad=False)
        #self.grid_search_variables(model_params, device)
        
    def forward(self, data):
        y, penultimate = self.model.penultimate(data)
        
        diff = penultimate[:,None,:] - self.means[None,:,:]
        
        vec2 = (self.precision[None,None,:,:] * diff[:,:,None,:]).sum(-1)
        dist = (diff * vec2).sum(-1).min(1)[0]
        
        idx = y.max(1)[1]
        idx = (self.class_vec==idx[:,None])
        out = -np.inf * torch.ones_like(y)
        out[idx] = -dist
        
        return out
    
    def sample_estimator(self, num_classes, feature_list, train_loader, device):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """
        import sklearn.covariance
        
        model = self.model

        model.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in train_loader:
            total += data.size(0)
            data = data.to(device)
            output, out_features = model.penultimate(data)
            out_features = [out_features]

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.to(device)).cpu()
            correct += equal_flag.sum()

            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse            
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        return sample_class_mean, precision

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_of_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_of_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        out = self.linear(out)
        return out
    
    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        penultimate = out.view(out.shape[0], -1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, penultimate


def ResNet18(num_of_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], 
                  num_of_channels=num_of_channels, 
                  num_classes=num_classes)


class LeNetMadry(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def penultimate(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
        penultimate = F.relu(self.fc1(x))
        out = self.fc2(penultimate)
        return out, penultimate


### Calibration Functions
from sklearn.metrics import roc_auc_score

def test_metrics(model, device, in_loader, out_loader):
    with torch.no_grad():
        model.eval()
        conf_in = []
        conf_out = []
        

        for i, (data_in, _) in enumerate(in_loader):
            data_in = data_in.to(device)
            out = model(data_in)
            output_in = out.max(1)[0]
            
         #   min_conf = 1./out.shape[1]
            
         #   idx = output_in < min_conf
         #   output_in[idx] = min_conf
            conf_in.append(output_in)
            if i>10:
                break;
            
        for i, (data_out, _) in enumerate(out_loader):    
            data_out = data_out.to(device)
            out = model(data_out)
            output_out = out.max(1)[0]
            
          #  min_conf = 1./out.shape[1]
          #  idx = output_out < min_conf
          #  output_out[idx] = min_conf
            conf_out.append(output_out)
            if i>10:
                break;
            
        conf_in = torch.cat(conf_in)
        conf_out = torch.cat(conf_out)
        
        y_true = torch.cat([torch.ones_like(conf_in), 
                            torch.zeros_like(conf_out)]).cpu().numpy()
        y_scores = torch.cat([conf_in, 
                              conf_out]).cpu().numpy()
        
        mmc = conf_out.exp().mean().item()
        auroc = roc_auc_score(y_true, y_scores)
        fp95 = ((conf_out.exp() > 0.95).float().mean().item())
        return mmc, auroc, fp95
