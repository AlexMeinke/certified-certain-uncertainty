import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetTemp(nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.temperature = temperature
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
        self.temperature = temperature
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
            losses.sum().backward()

        x = x + self.epsilon * x.grad
        x = torch.clamp(x, 0, 1).requires_grad_()
        return x
