import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import transform_train, transform_train_augmented, transform_test
from utils import network, optimizers, data_list

from algorithms.source_only.source_only import SourceOnly


class SourceOnlyAugmented(SourceOnly):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(SourceOnlyAugmented, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_dsets(self):
        self.dsets = {}
        root = os.path.join(self.dset_hp['root'], self.dset_hp['name'])
        self.dsets["source"] = data_list.ImageList(open(self.dset_hp['dset_path_train']).readlines(),
                                                   transform=transform_train_augmented(), root=root)

        self.dsets["test"] = data_list.ImageList(open(self.dset_hp['dset_path_val']).readlines(), 
                                                 transform=transform_test(), root=root)

