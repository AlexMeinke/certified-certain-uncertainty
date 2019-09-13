from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, cdist, squareform
import utils.dataloaders as dl

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


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
        
        self.scale_factor = 1.
        
        self.to(device)
        self.eval()
        
        self.grid_search_variables(model_params, device)
        
        
    def forward(self, data):
        conf = self.forward_all_layers(data).sum(0)
        
        out = self.model(data)
        idx = out.max(1)[1]
        idx = (self.class_vec==idx[:,None])
        out = -np.inf*torch.ones_like(out)
        out[idx] = conf / self.scale_factor
        return out
    
    def forward_all_layers(self, data):
        conf = []
        for i in range(len(self.sample_mean)):
            out = self.forward_layer(data, i)
            conf.append(out)
            
        conf = (torch.stack(conf, 0) * self.alpha[:,None])
        
        return conf
    
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
        
        #noise_gaussian_score *= (noise_gaussian_score < 0).float()
        
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
        #shape = next(iter(model_params.train_loader))[0][0].shape

        #epsilons = np.linspace(0, 0.004, 5)
        epsilons = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]

        vec = []
        alpha_vec = []
        for eps in epsilons:
            self.magnitude  = eps
            
            print('Logistic regression')
            alpha_vec.append(self.logistic_regression(model_params.train_loader, 
                                                      model_params.tinyimage_loader))
            
            print('Testing')
            _, auroc, fp95 = test_metrics(self, device, 
                                          model_params.train_loader,
                                          model_params.tinyimage_loader)
            
            print(auroc)
            vec.append([auroc, fp95])
        
        auroc = torch.tensor(vec)[:,0]
        fp95 = torch.tensor(vec)[:,1]
        
        
        ind = auroc.view(-1).argmax().item()

        alpha = alpha_vec[ind]
        self.alpha = nn.Parameter(alpha, requires_grad=False)

        self.magnitude  = epsilons[ind]
        
    def logistic_regression(self, in_loader, out_loader):
        self.eval()
        in_loader = iter(in_loader)
        out_loader = iter(out_loader)
        y = []
        target = []
        for _ in range(10):
            batch_in = next(in_loader)[0].to(self.device)
            batch_out = next(out_loader)[0].to(self.device)
            target.append(torch.cat([torch.ones(batch_in.shape[0]),
                                     torch.zeros(batch_out.shape[0])], 0))
                         
            batch = torch.cat([batch_in, batch_out], 0)
            conf = self.forward_all_layers(batch).detach().cpu()
            y.append(conf)
        y = torch.cat(y, 1).t()
        target = torch.cat(target, 0)
        
        clf = LogisticRegressionCV(random_state=0, solver='liblinear').fit(y, target)
        
        alpha = torch.tensor(clf.coef_, dtype=torch.float, device=self.device).squeeze(0)
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        return alpha


import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # F.log_softmax(out, dim=1)
        return out 

    
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
    
    
def ResNet18(num_of_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], 
                  num_of_channels=num_of_channels, 
                  num_classes=num_classes)

    
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
        #x = F.log_softmax(x, dim=1)
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
        x
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
