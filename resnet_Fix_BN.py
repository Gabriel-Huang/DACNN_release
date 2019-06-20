import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_Fix(nn.Module):

    def __init__(self, conv, in_planes, planes, stride=1):
        super(BasicBlock_Fix, self).__init__()
        self.conv1 = conv
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
       # out = F.relu(self.bn2(self.conv1(out)))
        out = self.bn3(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class resnet_Fix_BN(nn.Module):
    def __init__(self, num_classes=100):
        super(resnet_Fix_BN, self).__init__()
        num_blocks = [3,4,6,3]
        self.in_planes = 128

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_fix = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(128, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(128, num_blocks[3], stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size = 1, stride = 1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_Fix(self.conv_fix, self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.maxpool(out)
        out = self.layer2(out)
        out = self.maxpool(out)
        out = self.layer3(out)
        out = self.maxpool(out)
        out = self.layer4(out)
        out = self.maxpool(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = resnet_Fix_BN()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
