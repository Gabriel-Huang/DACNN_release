import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_Fix(nn.Module):

    def __init__(self, conv, in_planes, planes, trans = None):
        super(BasicBlock_Fix, self).__init__()
        self.conv = conv
        self.trans = trans
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.regulator = nn.Conv2d(planes,planes,kernel_size = 1, stride = 1, padding = 0, bias = False)
        if trans is not None:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes))
    def swap(self, x):
        head = x[:,0].unsqueeze(1)
        tail = x[:,1:]
        return torch.cat((tail,head),1)

    def forward(self, x):
        if self.trans is not None:
            out = self.trans(x)
        else:
            out = F.relu(self.bn1(self.conv(x)))
        out = self.swap(out)
        out = self.bn2(self.conv(out))
        out = self.swap(out)
        out += self.swap(self.swap(self.shortcut(x)))
        out = F.relu(out)
        return out

class resnet_mix(nn.Module):
    def __init__(self, num_classes=100):
        super(resnet_mix, self).__init__()
        num_blocks = [3,4,6,3]
        self.in_planes = 64
        #num_blocks = [2,2,2,2]

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                           nn.BatchNorm2d(64),
                           nn.ReLU(inplace=True))

        trans1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True))

        trans2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True))

        trans3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True))

        conv_fix_64 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        conv_fix_128 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        conv_fix_256 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        conv_fix_512 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv_fix = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(conv_fix_64,64, 64, None, num_blocks[0])
        self.layer2 = self._make_layer(conv_fix_128, 64,128, trans1, num_blocks[1])
        self.layer3 = self._make_layer(conv_fix_256, 128,256, trans2, num_blocks[2])
        self.layer4 = self._make_layer(conv_fix_512, 256,512, trans3, num_blocks[3])
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, conv_fix, in_planes, planes, trans, num_blocks):

        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(BasicBlock_Fix(conv_fix,in_planes, planes, trans = trans))
            else:
                layers.append(BasicBlock_Fix(conv_fix,in_planes,  planes, trans = None))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

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
    net = resnet_mix()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
