'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
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
        return F.log_softmax(out, dim=1)

    
class WideResNetBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout=0.5):
        super(WideResNetBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if in_planes!=planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 
                                      kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, N, k, num_classes=10, num_of_channels=3, dropout=0.5):
        super(WideResNet, self).__init__()
        widths = [k*v for v in [16, 32, 64]]
        
        self.in_planes = 16
        self.dropout = dropout

        self.conv1 = nn.Conv2d(num_of_channels, self.in_planes, 
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        self.layer1 = WideResNetBasicBlock(self.in_planes, 16, dropout=self.dropout)
        self.layer2 = self._make_layer(16, 16*k, N)
        self.layer3 = self._make_layer(16*k, 32*k, N)
        self.layer4 = self._make_layer(32*k, 64*k, N)
        
        self.linear = nn.Linear(576, num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks):
        planes = [in_planes] + [out_planes]*(num_blocks-1)
        layers = []
        for plane in planes:
            layers.append(WideResNetBasicBlock(plane, out_planes, dropout=self.dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=1)

    
def ResNet18(num_of_channels=3, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], 
                  num_of_channels=num_of_channels, 
                  num_classes=num_classes)

def ResNet18_100():
    return ResNet(BasicBlock, [2,2,2,2], num_classes=100)
