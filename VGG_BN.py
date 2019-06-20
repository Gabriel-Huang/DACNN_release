'''
DACNN14(plain) based on VGG.

'''

import torch
import torch.nn as nn

class vgg_Fix_BN(nn.Module):
    def __init__(self):
        super(vgg_Fix_BN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True))

        self.BN1 = nn.BatchNorm2d(128)
        self.BN2 = nn.BatchNorm2d(128)

        self.BN3 = nn.BatchNorm2d(128)
        self.BN4 = nn.BatchNorm2d(128)

        self.BN5 = nn.BatchNorm2d(128)
        self.BN6 = nn.BatchNorm2d(128)
        self.BN7 = nn.BatchNorm2d(128)

        self.BN8 = nn.BatchNorm2d(128)
        self.BN9 = nn.BatchNorm2d(128)
        self.BN10 = nn.BatchNorm2d(128)

        self.BN11 = nn.BatchNorm2d(128)
        self.BN12 = nn.BatchNorm2d(128)
        self.BN13 = nn.BatchNorm2d(128)

        self.AvgPool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(128, 100) # 128C, 10L

    def forward(self, x):
        out = self.conv1(x)
        out = self.BN1(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = self.MaxPool(out)

        out = self.conv2(out)
        out = self.BN3(out)
        out = self.conv2(out)
        out = self.BN4(out)
        out = self.MaxPool(out)

        out = self.conv2(out)
        out = self.BN5(out)
        out = self.conv2(out)
        out = self.BN6(out)
        out = self.conv2(out)
        out = self.BN7(out)
        out = self.MaxPool(out)

        out = self.conv2(out)
        out = self.BN8(out)
        out = self.conv2(out)
        out = self.BN9(out)
        out = self.conv2(out)
        out = self.BN10(out)
        out = self.MaxPool(out)

        out = self.conv2(out)
        out = self.BN11(out)
        out = self.conv2(out)
        out = self.BN12(out)
        out = self.conv2(out)
        out = self.BN13(out)
        out = self.MaxPool(out)

        out = self.AvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def test():
    net = vgg_Fix_BN()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
