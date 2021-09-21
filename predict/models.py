import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models import *

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine

class ResnetClass(nn.Module):
    def load_pretrained(self, pretrained):
        print('load model: %s' % pretrained)
        checkpoint = torch.load(pretrained)
        self.load_state_dict(checkpoint['state_dict'])

    def __init__(self,
                 backbone='resnet34',
                 num_classes=28,
                 in_channels=4,
                 pretrained=None,
                 ):
        super().__init__()

        if backbone == 'resnet18':
            self.resnet = resnet18()
            self.EX = 1
        elif backbone == 'resnet34':
            self.resnet = resnet34()
            self.EX = 1
        elif backbone == 'resnet50':
            self.resnet = resnet50()
            self.EX = 4
        elif backbone == 'resnet101':
            self.resnet = resnet101()
            self.EX = 4
        elif backbone == 'resnet152':
            self.resnet = resnet152()
            self.EX = 4
        else:
            raise ValueError(backbone)

        self.in_channels = in_channels
        if self.in_channels == 4:
            w = self.resnet.conv1.weight
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.resnet.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))

        self.encoder1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.bn1 = nn.BatchNorm1d(1024 * self.EX)
        self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
        self.bn2 = nn.BatchNorm1d(512 * self.EX)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512 * self.EX, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.logit = nn.Linear(512 * self.EX, num_classes)
        self.arc_margin_product = ArcMarginProduct(512, num_classes)

        self.load_pretrained(pretrained)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(x.size(0), -1)

        x = self.fc2(x)
        feature = self.bn3(x)

        return feature

def class_resnet50_dropout(num_classes=28, in_channels=4, pretrained=None):
    model = ResnetClass(
        backbone='resnet50',
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained
    )
    return model

class DensenetClass(nn.Module):
    def load_pretrained(self, pretrained):
        print('load model: %s' % pretrained)
        checkpoint = torch.load(pretrained)
        self.load_state_dict(checkpoint['state_dict'])

    def __init__(self,
                 backbone='densenet121',
                 num_classes=28,
                 in_channels=4,
                 pretrained=None,
                 large=False,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.large = large

        if backbone == 'densenet121':
            self.backbone = densenet121()
            num_features = 1024
        elif backbone == 'densenet169':
            self.backbone = densenet169()
            num_features = 1664
        elif backbone == 'densenet161':
            self.backbone = densenet161()
            num_features = 2208
        elif backbone == 'densenet201':
            self.backbone = densenet201()
            num_features = 1920
        else:
            raise ValueError(backbone)

        if self.in_channels == 4:
            w = self.backbone.features.conv0.weight
            self.backbone.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.backbone.features.conv0.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))

        self.conv1 =nn.Sequential(
            self.backbone.features.conv0,
            self.backbone.features.norm0,
            self.backbone.features.relu0,
            self.backbone.features.pool0
        )
        self.encoder2 = nn.Sequential(
            self.backbone.features.denseblock1,
        )
        self.encoder3 = nn.Sequential(
            self.backbone.features.transition1,
            self.backbone.features.denseblock2,
        )
        self.encoder4 = nn.Sequential(
            self.backbone.features.transition2,
            self.backbone.features.denseblock3,
        )
        self.encoder5 = nn.Sequential(
            self.backbone.features.transition3,
            self.backbone.features.denseblock4,
            self.backbone.features.norm5
        )
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.bn1 = nn.BatchNorm1d(num_features*2)
        self.fc1 = nn.Linear(num_features*2, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)
        self.relu = nn.ReLU(inplace=True)

        self.logit = nn.Linear(num_features, num_classes)

        self.load_pretrained(pretrained)

    def forward(self, x):
        mean = [0.074598, 0.050630, 0.050891, 0.076287] # rgby
        std  = [0.122813, 0.085745, 0.129882, 0.119411]
        for i in range(self.in_channels):
            x[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]

        x = self.conv1(x)
        if self.large:
            x = self.maxpool(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e5 = F.relu(e5, inplace=True)

        x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        feature = x.view(x.size(0), -1)
        x = self.logit(feature)
        return x, feature

def class_densenet121_dropout(num_classes=28, in_channels=4, pretrained=None):
    model = DensenetClass(
        backbone='densenet121',
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained
    )
    return model

def class_densenet121_large_dropout(num_classes=28, in_channels=4, pretrained=None):
    model = DensenetClass(
        backbone='densenet121',
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        large=True
    )
    return model
