import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import transform_train, transform_test, transforms_10crop
from utils import network, optimizers, data_list

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Algorithm, self).__init__()
        self.dset_hp = dset_hp
        self.loss_hp = loss_hp
        self.train_hp = train_hp
        self.net_hp = net_hp
        self.logger_hp = logger_hp
        if 'radius' in loss_hp.keys():
            net_hp['radius'] = loss_hp['radius']

    def set_dsets(self):
        self.dsets = {}
        root = os.path.join(self.dset_hp['root'], self.dset_hp['name'])
        self.dsets['source'] = data_list.ImageList(open(self.dset_hp['s_dset_path']).readlines(),
                                                   transform=transform_train(), root=root)
        self.dsets['target'] = data_list.ImageList(open(self.dset_hp['t_dset_path']).readlines(),
                                                   transform=transform_train(), root=root)
        self.dsets['test'] = data_list.ImageList(open(self.dset_hp['t_dset_path']).readlines(), 
                                                 transform=transform_test(), root=root)
    def set_dsets_model_selection(self):
        if self.dset_hp['use_val']:
            root = os.path.join(self.dset_hp['root'], self.dset_hp['name'])
            self.dsets['source_train'] = data_list.ImageList(open(self.dset_hp['s_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)
            self.dsets['source_val'] = data_list.ImageList(open(self.dset_hp['v_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)
        if '1shot_acc' in self.logger_hp['model_selection_metrics']:
            self.dsets['test_1shot'] = data_list.ImageList(open(self.dset_hp['t_1shot_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)            
        if '1shot_10crop_acc' in self.logger_hp['model_selection_metrics']:
            list_transforms_10crop = transforms_10crop()
            self.dsets['test_1shot_10crop'] = {}
            for i in range(10):
                self.dsets['test_1shot_10crop'][i] = data_list.ImageList(open(self.dset_hp['t_1shot_dset_path']).readlines(),
                                                   transform=list_transforms_10crop[i], root=root)
        if '3shot_acc' in self.logger_hp['model_selection_metrics']:
            self.dsets['test_3shot'] = data_list.ImageList(open(self.dset_hp['t_3shot_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)
        if '3shot_10crop_acc' in self.logger_hp['model_selection_metrics']:
            list_transforms_10crop = transforms_10crop()
            self.dsets['test_3shot_10crop'] = {}
            for i in range(10):
                self.dsets['test_3shot_10crop'][i] = data_list.ImageList(open(self.dset_hp['t_3shot_dset_path']).readlines(),
                                                   transform=list_transforms_10crop[i], root=root)
        if '25random_acc' in self.logger_hp['model_selection_metrics']:
            self.dsets['test_25random'] = data_list.ImageList(open(self.dset_hp['t_25random_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)            
        if '25random_10crop_acc' in self.logger_hp['model_selection_metrics']:
            list_transforms_10crop = transforms_10crop()
            self.dsets['test_25random_10crop'] = {}
            for i in range(10):
                self.dsets['test_25random_10crop'][i] = \
                    data_list.ImageList(open(self.dset_hp['t_25random_dset_path']).readlines(),
                                                   transform=list_transforms_10crop[i], root=root)
        if '50random_acc' in self.logger_hp['model_selection_metrics']:
            self.dsets['test_50random'] = data_list.ImageList(open(self.dset_hp['t_50random_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)            
        if '50random_10crop_acc' in self.logger_hp['model_selection_metrics']:
            list_transforms_10crop = transforms_10crop()
            self.dsets['test_50random_10crop'] = {}
            for i in range(10):
                self.dsets['test_50random_10crop'][i] = \
                    data_list.ImageList(open(self.dset_hp['t_50random_dset_path']).readlines(),
                                                   transform=list_transforms_10crop[i], root=root)
        if '100random_acc' in self.logger_hp['model_selection_metrics']:
            self.dsets['test_100random'] = data_list.ImageList(open(self.dset_hp['t_100random_dset_path']).readlines(),
                                                   transform=transform_test(), root=root)            
        if '100random_10crop_acc' in self.logger_hp['model_selection_metrics']:
            list_transforms_10crop = transforms_10crop()
            self.dsets['test_100random_10crop'] = {}
            for i in range(10):
                self.dsets['test_100random_10crop'][i] = \
                    data_list.ImageList(open(self.dset_hp['t_100random_dset_path']).readlines(),
                                                   transform=list_transforms_10crop[i], root=root)


    def set_dset_loaders(self):
        self.dset_loaders = {}
        self.dset_loaders['source'] = DataLoader(self.dsets['source'],
                                        batch_size=self.train_hp['train_bs'],
                                        shuffle=True, 
                                        num_workers=self.train_hp['num_workers'], 
                                        drop_last=True)
        self.dset_loaders['target'] = DataLoader(self.dsets['target'],
                                                 batch_size=self.train_hp['train_bs'],
                                                 shuffle=True,
                                                 num_workers=self.train_hp['num_workers'],
                                                 drop_last=True)
        self.dset_loaders['test']   = DataLoader(self.dsets['test'],
                                                 batch_size=self.train_hp['test_bs'],
                                                 shuffle=False,
                                                 num_workers=self.train_hp['num_workers'])
    def set_dset_loaders_model_selection(self):
        if self.dset_hp['use_val']:
            self.dset_loaders['source_train'] = DataLoader(self.dsets['source_train'],
                                            batch_size=self.train_hp['test_bs'],
                                            shuffle=True, 
                                            num_workers=self.train_hp['num_workers'], 
                                            drop_last=False)
            self.dset_loaders['source_val'] = DataLoader(self.dsets['source_val'],
                                            batch_size=self.train_hp['test_bs'],
                                            shuffle=False, 
                                            num_workers=self.train_hp['num_workers'], 
                                            drop_last=False)
        for dset in self.dsets:
            if ('shot' in dset) or ('random' in dset):
                if '10crop' in dset:
                    self.dset_loaders[dset] = {}
                    for i in range(10):
                        self.dset_loaders[dset][i] = DataLoader(self.dsets[dset][i],
                                                batch_size=self.train_hp['test_bs'],
                                                shuffle=False, 
                                                num_workers=self.train_hp['num_workers'], 
                                                drop_last=False)
                else:
                    self.dset_loaders[dset] = DataLoader(self.dsets[dset],
                                            batch_size=self.train_hp['test_bs'],
                                            shuffle=False, 
                                            num_workers=self.train_hp['num_workers'], 
                                            drop_last=False)

    def set_base_network(self):
        self.base_network = network.get_base_network(self.net_hp)

    def prep_for_train(self):
        self.parameter_list = self.base_network.get_parameters()
        self.base_network = self.base_network.cuda()
        if self.net_hp['load_net']:
            self.base_network.load_state_dict(torch.load(os.path.join(self.net_hp['load_path'])))
        self.base_network = torch.nn.DataParallel(self.base_network).cuda()
        if self.train_hp['optimizer'] == 'default':
            self.optimizer, self.schedule_param, self.lr_scheduler = optimizers.set_default_optimizer_scheduler(
                self.train_hp, self.parameter_list)

    def update_dsets(self, i):
        pass
    
    def update_dset_loaders(self, i):
        # Restart iterators if needed
        if i % len(self.dset_loaders['source']) == 0:
            self.iter_source = iter(self.dset_loaders['source'])
        if i % len(self.dset_loaders['target']) == 0:
            self.iter_target = iter(self.dset_loaders['target'])
                         
    def save_model(self, name):
        torch.save(self.base_network.module.state_dict(), os.path.join(self.logger_hp['output_dir'], name))
                         
    def update(self):
        """
        Computes the loss and performs one update step.
        """
        raise NotImplementedError
        
