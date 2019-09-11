from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, cdist, squareform



### Mahalanobis Detector

class Mahalanobis(nn.Module):
    def __init__(self, model, model_params, device=torch.device('cpu')):
        super().__init__()
        self.model = model
        
        self.device = device
        
        model.eval()
        temp_x = next(iter(model_params.train_loader))[0].to(device)
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        
        sample_mean, precision = self.sample_estimator(model, model_params.classes, 
                                                       feature_list, model_params.train_loader)
        for i, mean in enumerate(sample_mean):
            sample_mean[i] = nn.Parameter(mean, requires_grad=False)
        self.sample_mean = nn.ParameterList(sample_mean)
        
        for i, pre in enumerate(precision):
            precision[i] = nn.Parameter(pre, requires_grad=False)
        self.precision = nn.ParameterList(precision)
        
        self.num_classes = model_params.classes
        
        
        self.magnitude = 0.
        self.alpha = nn.Parameter(torch.ones(num_output), requires_grad=False)
        
        self.class_vec = nn.Parameter(torch.arange(self.num_classes)[None,:], requires_grad=False)
        
        self.scale_factor = 500.
        
        self.to(device)
        self.grid_search_variables(model_params, device)
        
    def forward(self, data):
        data
        conf = []
        for i in range(len(self.sample_mean)):
            out = self.forward_layer(data, i)
            conf.append(out)
            
        conf = (torch.stack(conf, 0) * self.alpha[:,None]).sum(0)
        
        out = self.model(data)
        idx = out.max(1)[1]
        idx = (self.class_vec==idx[:,None])
        out = -np.inf*torch.ones_like(out)
        out[idx] = conf / self.scale_factor
        return out
    
    def forward_layer(self, data, layer_index):
        with torch.enable_grad():
            # data = data.clone().requires_grad_()
            data = data.requires_grad_()
            out_features = self.model.intermediate_forward(data, layer_index)

            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            
            
            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(self.num_classes):
                batch_sample_mean = self.sample_mean[layer_index][i]
                zero_f = out_features - batch_sample_mean

                term_gau = -0.5*torch.mm(zero_f @ self.precision[layer_index], zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = self.sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - batch_sample_mean
            pure_gau = -0.5*torch.mm(zero_f @ self.precision[layer_index], zero_f.t()).diag()

            loss = -pure_gau.mean()

            gradient = torch.autograd.grad (loss, data)[0]
            

        tempInputs = (data - self.magnitude * gradient).requires_grad_()

        noise_out_features = self.model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), 
                                                     noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)

        noise_gaussian_score = 0
        for i in range(self.num_classes):
            batch_sample_mean = self.sample_mean[layer_index][i]
            zero_f = noise_out_features - batch_sample_mean
            term_gau = -0.5*torch.mm(zero_f @ self.precision[layer_index], zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        
        tempInputs = data + self.magnitude * gradient
        
        
        return noise_gaussian_score
        
    def sample_estimator(self, model, num_classes, feature_list, train_loader):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """
        import sklearn.covariance

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
            data = data.to(self.device).requires_grad_()
            output, out_features = model.feature_list(data)

            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.to(self.device)).cpu()
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
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(self.device)

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
            temp_precision = torch.from_numpy(temp_precision).float().to(self.device)
            precision.append(temp_precision)

        #print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision
    
    def grid_search_variables(self, model_params, device):
        shape = next(iter(model_params.train_loader))[0][0].shape
        temperatures = [1]
        epsilons = np.linspace(0, 0.004, 5)

        grid = []

        for T in temperatures:
            vec = []
            for eps in epsilons:
                self.magnitude  = eps
                #stats = aggregate_stats([self], device, shape, 
                #                        classes=self.num_classes)
                _, auroc, fp95 = test_metrics(self, device, 
                                              model_params.train_loader,
                                              model_params.loaders[0][1])
                #print(str(eps)+ ': ' + str(auroc))
                vec.append([auroc, fp95])
            grid.append(vec)
        
        auroc = torch.tensor(grid)[:,:,0]
        fp95 = torch.tensor(grid)[:,:,1]

        ind = auroc.view(-1).argmax().item()

        xv, yv = np.meshgrid(epsilons, temperatures)
        T = yv.reshape(-1)[ind]
        eps = xv.reshape(-1)[ind]

        self.magnitude  = eps


### Model Architecture
    
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
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(in_channels,64)
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
        return F.log_softmax(y, dim=1)
    
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
        return y, penultimate
    
def ResNet18(num_c, in_channels=3):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_c, in_channels=in_channels)

def ResNet34(num_c, in_channels=3):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c, in_channels=in_channels)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

    
class LeNet(nn.Module):
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
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        if layer_index == 0:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
        elif layer_index == 1:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
        elif layer_index == 2:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 7*7*64)
            x = self.fc1(x)
        elif layer_index == 3:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 7*7*64)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)         
        return x
    
    def feature_list(self, x):
        out_list = []
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        out_list.append(x)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)       
        out_list.append(x)
        
        x = x.view(-1, 7*7*64)
        x = self.fc1(x)
        out_list.append(x)
        
        x = self.fc2(F.relu(x))
        out_list.append(x)
        x = F.log_softmax(x, dim=1)
        return x, out_list

    
### Calibration Functions
from sklearn.metrics import roc_auc_score

def get_auroc(model_list, loader, stats, device):
    auroc = []
    fp95 = []
    for i, model in enumerate(model_list):
        with torch.no_grad():
            conf = []
            for data, _ in loader:
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


def test_metrics(model, device, in_loader, out_loader):
    with torch.no_grad():
        model.eval()
        conf_in = []
        conf_out = []
        

        for data_in, _ in in_loader:
            data_in = data_in.to(device)
            out = model(data_in)
            output_in = out.max(1)[0]
            
         #   min_conf = 1./out.shape[1]
            
         #   idx = output_in < min_conf
         #   output_in[idx] = min_conf
            conf_in.append(output_in)
            
        for data_out, _ in out_loader:    
            data_out = data_out.to(device)
            out = model(data_out)
            output_out = out.max(1)[0]
            
          #  min_conf = 1./out.shape[1]
          #  idx = output_out < min_conf
          #  output_out[idx] = min_conf
            conf_out.append(output_out)
            
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
