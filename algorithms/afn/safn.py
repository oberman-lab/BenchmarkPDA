import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import transform_train, transform_test
from utils import network, optimizers, data_list
from utils import optimizers, data_list
# import algorithms.afn.network as network # uncomment to use the network described in the SAFN paper

from algorithms.base_algorithm import Algorithm

import algorithms.afn.utils as afn_utils

class Safn(Algorithm):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Safn, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

#     def set_base_network(self): # uncomment to use the network described in the SAFN paper
#         self.base_network = network.get_base_network(self.net_hp)

    def update(self, i):
        """
        Computes the loss and performs one update step.
        """
        
        self.base_network.train(True)
        xs, ys = self.iter_source.next()
        xt, _ = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        g_xs, f_g_xs = self.base_network(xs)
        g_xt, f_g_xt = self.base_network(xt)
        
        clf_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys)
        
        L2norm_loss_s = afn_utils.get_L2norm_loss_self_driven(g_xs, self.loss_hp['lambda'])
        L2norm_loss_t = afn_utils.get_L2norm_loss_self_driven(g_xt, self.loss_hp['delta_r'])

        adpt_loss = self.loss_hp['lambda'] * (L2norm_loss_s + L2norm_loss_t)
        
        total_loss = clf_loss + adpt_loss 
        
        # Updating weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, clf_loss, adpt_loss