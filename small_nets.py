'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    '3_layer': ['c', 'M','c','M'],
    '5_layer': ['c', 'M', 'c', 'M', 'c', 'M', 'c', 'M'],
    '7_layer': ['c',' M', 'c', 'M', 'c', 'c', 'M', 'c', 'c','M'],
    '9_layer': ['c', 'c',' M', 'c', 'c', 'M', 'c', 'c', 'M', 'c', 'c', 'M'],
    '11_layer': ['c', 'c',' M', 'c', 'c', 'M', 'c', 'c', 'c', 'M', 'c', 'c', 'c', 'M'],


    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Small_Nets(nn.Module):
    def __init__(self, layers, fixed = False):
        super(Small_Nets, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True))

        self.conv_fix = [nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True)]

        self.features = self._make_layers(cfg[layers], conv_fix, fixed)
        self.classifier = nn.Linear(128, 100)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, conv_fix, fixed):
        layers = []
        if fixed:
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += conv_fix
        else:
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(inplace=True)]

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = Small_Nets('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
