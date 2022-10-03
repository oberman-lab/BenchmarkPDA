import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import transform_train, transform_test
from utils import network, optimizers, data_list

from algorithms.base_algorithm import Algorithm


class SourceOnly(Algorithm):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(SourceOnly, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_dsets(self):
        self.dsets = {}
        root = os.path.join(self.dset_hp['root'], self.dset_hp['name'])
        self.dsets["source"] = data_list.ImageList(open(self.dset_hp['dset_path_train']).readlines(),
                                                   transform=transform_train(), root=root)

        self.dsets["test"] = data_list.ImageList(open(self.dset_hp['dset_path_val']).readlines(), 
                                                 transform=transform_test(), root=root)

    def set_dset_loaders(self):        
        self.dset_loaders = {}
        self.dset_loaders["source"] = DataLoader(self.dsets["source"],
                                        batch_size=self.train_hp['train_bs'],
                                        shuffle=True, 
                                        num_workers=self.train_hp['num_workers'], 
                                        drop_last=True)
        self.dset_loaders["test"]   = DataLoader(self.dsets["test"],
                                                 batch_size=self.train_hp['test_bs'],
                                                 shuffle=False,
                                                 num_workers=self.train_hp['num_workers'])

    def prep_for_train(self):
        self.parameter_list = self.base_network.get_parameters(mode=self.train_hp['update'])
        self.base_network = torch.nn.DataParallel(self.base_network).cuda()
        if self.train_hp['optimizer'] == 'default':
            self.optimizer, self.schedule_param, self.lr_scheduler = optimizers.set_default_optimizer_scheduler(
                self.train_hp, self.parameter_list)

    def update_dset_loaders(self, i):
        # Restart iterators if needed
        if i % len(self.dset_loaders["source"]) == 0:
            self.iter_source = iter(self.dset_loaders["source"])                         
                         
    def update(self, i):
        """
        Computes the loss and performs one update step.
        """
        self.base_network.train(True)
        xs, ys = self.iter_source.next()
        xs, ys = xs.cuda(), ys.cuda()

        g_xs, f_g_xs = self.base_network(xs)

        loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys)

        # Updating weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, loss, torch.tensor(0.0)
