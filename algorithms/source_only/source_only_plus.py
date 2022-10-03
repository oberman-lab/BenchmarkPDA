import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import transform_train, transform_test
from utils import network, optimizers, data_list

from algorithms.base_algorithm import Algorithm


class SourceOnlyPlus(Algorithm):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(SourceOnlyPlus, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)
                         
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
