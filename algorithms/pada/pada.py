import os
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from utils import data_list, network, optimizers
from utils.datasets import transform_train, transform_test

from algorithms.base_algorithm import Algorithm
import algorithms.pada.utils as pada_utils

class Pada(Algorithm):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Pada, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_base_network(self):
        self.base_network = network.get_base_network(self.net_hp)
        self.ad_net = pada_utils.AdversarialNetwork(self.base_network.output_num())
        self.gradient_reverse_layer = pada_utils.AdversarialLayer(high_value=1.0)#config["high"])

    def prep_for_train(self):
        self.parameter_list = self.base_network.get_parameters() + self.ad_net.get_parameters()
        self.base_network = self.base_network.cuda()
        self.ad_net = self.ad_net.cuda()
        if self.net_hp['load_net']:
            self.base_network.load_state_dict(torch.load(os.path.join(self.net_hp['load_path'])))
        self.base_network = torch.nn.DataParallel(self.base_network).cuda()
        if self.train_hp['optimizer'] == 'default':
            self.optimizer, self.schedule_param, self.lr_scheduler = optimizers.set_default_optimizer_scheduler(
                self.train_hp, self.parameter_list)
        self.class_weight = torch.ones((1,self.dset_hp['class_num']),dtype=float).cuda()
        
    def update_dsets(self, i):
        if i % self.train_hp['weight_update_interval'] == 0:
            self.class_weight = pada_utils.get_class_weight(self.dset_loaders['test'], self.base_network, T=1.0).cuda()

    def update(self, i):
        """
        Computes the loss and performs one update step.
        """
        self.base_network.train(True)
        xs, ys = self.iter_source.next()
        xt, _, = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        
        inputs = torch.cat((xs, xt), dim=0)
        features, outputs = self.base_network(inputs)
        
        softmax_outputs = torch.nn.Softmax(dim=1)(outputs).detach()
        self.ad_net.train(True)
        weight_ad = torch.zeros(inputs.size(0))
        label_numpy = ys.data.cpu().numpy()
        for j in range(inputs.size(0) // 2):
            weight_ad[j] = self.class_weight[int(label_numpy[j])]

        weight_ad = weight_ad / torch.max(weight_ad[0:inputs.size(0)//2])
        for j in range(inputs.size(0) // 2, inputs.size(0)):
            weight_ad[j] = 1.0            
        transfer_loss = pada_utils.PADA(i, features, self.ad_net, self.gradient_reverse_layer, weight_ad)

        clf_loss = torch.nn.CrossEntropyLoss(weight=self.class_weight)(outputs.narrow(0, 0, inputs.size(0)//2), ys)
        adpt_loss = self.loss_hp['lambda'] * transfer_loss
        total_loss = clf_loss + adpt_loss
#         print(f'i = {i} Loss: {total_loss:.2f} = {clf_loss:.2f} + {adpt_loss:.2f}')
        # Updating weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss, clf_loss, adpt_loss
