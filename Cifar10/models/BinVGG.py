import collections
import sys

import math
import torch.nn as nn

sys.path.append("..")

from util import BinConv2d

cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class BinVGG(nn.Module):
    def __init__(self, vgg_name, is_train=True):
        super(BinVGG, self).__init__()
        self.is_train = is_train
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        if is_train:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = collections.OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
        ])
        in_channels = 64
        cnt = 1
        for x in cfg:
            if x == 'M':
                layers['pool' + str(cnt)] = nn.MaxPool2d(kernel_size=2, stride=2)
                cnt += 1
            else:
                layers['conv' + str(cnt)] = BinConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1,
                                                      is_train=self.is_train)
                cnt += 1
                layers['bn' + str(cnt)] = nn.BatchNorm2d(x)
                cnt += 1
                layers['relu' + str(cnt)] = nn.ReLU(inplace=True)
                cnt += 1
                in_channels = x
        layers['pool' + str(cnt)] = nn.AvgPool2d(kernel_size=1, stride=1)
        return nn.Sequential(layers)


class GroupedBinVGG(nn.Module):
    def __init__(self, vgg_name, is_train=True):
        super(GroupedBinVGG, self).__init__()
        self.is_train = is_train
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        if is_train:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = collections.OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
        ])
        in_channels = 64
        cnt = 1
        for x in cfg:
            if x == 'M':
                layers['pool' + str(cnt)] = nn.MaxPool2d(kernel_size=2, stride=2)
                cnt += 1
            else:
                groups = 2 if in_channels % 2 == 0 else 1

                layers['conv' + str(cnt)] = BinConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1,
                                                      is_train=self.is_train, groups=groups)
                cnt += 1
                layers['bn' + str(cnt)] = nn.BatchNorm2d(x)
                cnt += 1
                layers['relu' + str(cnt)] = nn.ReLU(inplace=True)
                cnt += 1
                in_channels = x
        layers['pool' + str(cnt)] = nn.AvgPool2d(kernel_size=1, stride=1)
        return nn.Sequential(layers)
