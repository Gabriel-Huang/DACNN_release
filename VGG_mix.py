'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    '3_layer': ['c', 'M','c','M'],
    '5_layer': ['c', 'M', 'c', 'M', 'c', 'M', 'c', 'M'],
    '7_layer': ['c',' M', 'c', 'M', 'c', 'c', 'M', 'c', 'c','M'],
    '9_layer': ['c', 'c',' M', 'c', 'c', 'M', 'c', 'c', 'M', 'c', 'c', 'M'],
    '11_layer': ['c', 'c',' M', 'c', 'c', 'M', 'c', 'c', 'c', 'M', 'c', 'c', 'c', 'M'],
    'VGG16_fix': [64, 64, 'M', 128, 128, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Mix_Net(nn.Module):
    def __init__(self, layers, fixed = False):
        super(Mix_Net, self).__init__()

        trans1 = [nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   nn.ReLU(inplace=True)]

        conv_fix_256 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        trans2 = [nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   nn.ReLU(inplace=True)]

        conv_fix_512 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.block_256 = self.make_block(256, trans1, conv_fix_256, 2)
        self.block_512 = self.make_block(512, trans2, conv_fix_512, 2)
        self.block_512_2 = self.make_block(512, [conv_fix_512], conv_fix_512, 2)

        self.features = self._make_layers(cfg[layers])
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = self.block_256(out)
        out = self.block_512(out)
        out = self.block_512_2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def make_block(self,channel, trans, fixconv, layer):
        layers = []
        layers+=trans
        for l in range(layer):
            layers+=[fixconv, nn.BatchNorm2d(channel),nn.ReLU(inplace=True)]
        layers+=[nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


def test():
    net = Mix_Net('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
