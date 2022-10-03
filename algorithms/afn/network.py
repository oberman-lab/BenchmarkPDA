import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F
import math

net_dict = {
    "ResNet18":models.resnet18,
    "ResNet34":models.resnet34,
    "ResNet50":models.resnet50,
    "ResNet101":models.resnet101,
    "ResNet152":models.resnet152,
    "ConvNextTiny":models.convnext_tiny,
    "ConvNextSmall":models.convnext_small,
    "ConvNextBase":models.convnext_base,
    "ConvNextLarge":models.convnext_large
}

def get_base_network(net_hparams: dict):
    if "ResNet" in net_hparams['net']:
        model = net_dict[net_hparams['net']](pretrained=True)
        feature_layers = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, \
                             model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool)
        in_features = model.fc.in_features
        base_network = BaseNetwork(feature_layers, in_features, net_hparams)

    if "ConvNext" in net_hparams['net']:
        params = {"convnext_name":net_hparams['net'],
                  "use_bottleneck":True,
                  "bottleneck_dim":net_hparams['bottleneck_dim'],
                  'class_num': net_hparams['class_num'],
                  "radius":net_hparams['radius'],
                  "use_slr":net_hparams['use_slr']
                 }
        model = net_dict[net_hparams['net']](pretrained=True)
        feature_layers = torch.nn.Sequential(model.features, model.avgpool, model.classifier[:2])
        in_features = model.classifier[-1].in_features
        base_network = BaseNetwork(feature_layers, in_features, net_hparams)

    return base_network

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class BaseNetwork(nn.Module):
    def __init__(self,
                 feature_layers: torch.nn.Sequential,
                 in_features: int,
                 net_hparams: dict):
        super(BaseNetwork, self).__init__()
        self.feature_layers = feature_layers
        self.use_bottleneck = True
        dropout_p = 0.5
        self.dropout_p = dropout_p
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
            )
        self.fc = nn.Linear(1000, net_hparams['class_num'])
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
        self.__in_features = 1000

        

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        if self.training:
            x.mul_(math.sqrt(1-self.dropout_p))
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self, mode='full'):
        if mode == 'clf_head':
            if self.use_bottleneck:
                parameter_list = [{"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
            else:
                parameter_list = [{"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]        
        elif mode == 'full':
            if self.use_bottleneck:
                parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                                {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
            else:
                parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                                {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        return parameter_list




