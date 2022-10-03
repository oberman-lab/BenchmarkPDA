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
        if net_hparams['nonlinear']:
            base_network = BaseNetworkNonLinear(feature_layers, in_features, net_hparams)
        else:
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
        self.use_bottleneck = net_hparams['use_bottleneck']
        self.use_slr = net_hparams['use_slr']
        self.radius = net_hparams['radius']

        if self.use_bottleneck:
            if self.use_slr:
                self.bottleneck = nn.Linear(in_features, net_hparams['bottleneck_dim'])
                self.bottleneck.apply(init_weights)
                self.fc = SLR_layer(bottleneck_dim, net_hparams['class_num'], bias=True)
                self.__in_features = net_hparams['bottleneck_dim']
            else:
                self.bottleneck = nn.Linear(in_features, net_hparams['bottleneck_dim'])
                self.fc = nn.Linear(net_hparams['bottleneck_dim'], net_hparams['class_num'])
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = net_hparams['bottleneck_dim']
        else:
            if self.use_slr:
                self.fc = SLR_layer(net_hparams['bottleneck_dim'], net_hparams['class_num'], bias=True)
            else:
                self.fc = nn.Linear(in_features, net_hparams['class_num'])
                self.fc.apply(init_weights)
            self.__in_features = in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        if self.use_slr or self.radius>0:
            x = self.radius*F.normalize(x, dim=1)
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


# Network Architecture used in the SAFN. The bottleneck layer is nonlinear with dropout
class BaseNetworkNonLinear(nn.Module):
    def __init__(self,
                 feature_layers: torch.nn.Sequential,
                 in_features: int,
                 net_hparams: dict):
        super(BaseNetworkNonLinear, self).__init__()
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
    
    
    
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, max_iter = 10000):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
    
# https://github.com/XJTU-XGU/Adversarial-Reweighting-for-Partial-Domain-Adaptation
class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features,bias=True):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias=torch.nn.Parameter(torch.zeros(out_features))
        self.bias_bool = bias
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r=input.norm(dim=1).detach()[0]
        if self.bias_bool:
            cosine = F.linear(input, F.normalize(self.weight),r*torch.tanh(self.bias))
        else:
            cosine = F.linear(input, F.normalize(self.weight))
        output=cosine
        return output

class WassersteinDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(WassersteinDiscriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]

    def grl_hook(coeff):
        def fun1(grad):
            return -coeff*grad.clone()
        return fun1

    def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
        return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


