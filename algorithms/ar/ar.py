import os
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from utils import data_list, network, optimizers
from utils.datasets import transform_train, transform_test

import utils.model_selection

from algorithms.base_algorithm import Algorithm
import algorithms.ar.utils as ar_utils
import algorithms.ar.get_weight as ar_get_weight
import copy

class Ar(Algorithm):
    """
    A subclass of Algorithm implements a partial domain adaptation algorithm.
    Subclasses should implement the following:
    - update()
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Ar, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_dsets(self):
        root = os.path.join(self.dset_hp['root'], self.dset_hp['name'])
        if self.train_hp['sampler'] == "subset_sampler":
            self.source_base_dataset_train = data_list.ImageList(open(self.dset_hp['s_dset_path']).readlines(),
                                                            transform=image_train(), root=root)
            self.source_base_dataset_test = data_list.ImageList(open(self.dset_hp['s_dset_path']).readlines(),
                                                           transform=image_test(), root=root)
        self.dsets = {}
        self.dsets["source"] = data_list.ImageList(open(self.dset_hp['s_dset_path']).readlines(),
                                                   transform=transform_train(), return_index=True, root=root)
        self.dsets["target"] = data_list.ImageList(open(self.dset_hp['t_dset_path']).readlines(),
                                                   transform=transform_train(), return_index=True, root=root)
        self.dsets["test"] = data_list.ImageList(open(self.dset_hp['t_dset_path']).readlines(), 
                                                 transform=transform_test(), root=root)
        self.dsets["source_aux"] = data_list.ImageList(open(self.dset_hp['s_dset_path']).readlines(), 
                                                 transform=transform_test(), root=root)

    def set_dset_loaders(self):
        self.dset_loaders = {}
        self.dset_loaders["source"] = DataLoader(self.dsets["source"],
                                        batch_size=self.train_hp['train_bs'],
                                        shuffle=True, 
                                        num_workers=self.train_hp['num_workers'], 
                                        drop_last=True)
        self.dset_loaders["target"] = DataLoader(self.dsets["target"], 
                                                 batch_size=self.train_hp['train_bs'],
                                                 shuffle=True,
                                                 num_workers=self.train_hp['num_workers'],
                                                 drop_last=True)
        self.dset_loaders["test"]   = DataLoader(self.dsets["test"],
                                                 batch_size=self.train_hp['test_bs'],
                                                 shuffle=False,
                                                 num_workers=self.train_hp['num_workers'])
        self.dset_loaders['source_aux'] = DataLoader(self.dsets['source_aux'],
                                                     batch_size=self.train_hp['test_bs'], shuffle=False,
                                                     num_workers=self.train_hp['num_workers'])

    def prep_for_train(self):
        self.parameter_list = self.base_network.get_parameters()
        self.base_network = self.base_network.cuda()
        if self.net_hp['load_net']:
            self.base_network.load_state_dict(torch.load(os.path.join(self.net_hp['load_path'])))
        if self.train_hp['optimizer'] == 'default':
            self.optimizer, self.schedule_param, self.lr_scheduler = optimizers.set_default_optimizer_scheduler(
                self.train_hp, self.parameter_list)
        self.weights = None

    def update_dsets(self, i):
        ## update weight, loader
        if self.train_hp['sampler'] == "weighted_sampler":
            if self.dset_hp['name'] == "domainnet" :
                seed = None
            else:
                seed = self.train_hp['seed']
            if i % self.train_hp['weight_update_interval'] == 0 and i>0:
                self.base_network.train(False)
                all_source_features, _, _ = ar_utils.get_features(self.dset_loaders['source_aux'], self.base_network)
                all_target_features, _, _ = ar_utils.get_features(self.dset_loaders['test'], self.base_network)
                self.weights = ar_get_weight.get_weight(
                    all_source_features,
                    all_target_features, 
                    self.loss_hp['rho0'],
                    seed,
                    self.train_hp['max_iter_discriminator'],
                    self.train_hp['automatical_adjust'],
                    self.loss_hp['up'],
                    self.loss_hp['low'],
                    i,
                    self.train_hp['multiprocess'],
                    self.loss_hp['c'])
                self.weights = torch.Tensor(self.weights[:])
                self.dset_loaders['source'] = DataLoader(self.dsets['source'],
                                                    batch_size=self.train_hp['train_bs'],
                                                    sampler=WeightedRandomSampler(self.weights,
                                                                                  num_samples=len(self.weights),
                                                                                  replacement=True),
                                                    num_workers=self.train_hp['num_workers'], drop_last=True)
        if self.train_hp['sampler'] == 'subset_sampler':
            if i % self.train_hp['weight_update_interval'] == 0 and i > 5000:
                indexes = np.random.permutation(len(source_base_dataset_test))[:args.train_bs * 2000]
                self.dsets['source'] = data_list.SubDataset(self.source_base_dataset_train, indexes)
                self.dsets['source_aux'] = data_list.SubDataset(self.source_base_dataset_test, indexes)
                self.dset_loaders['source_aux'] = DataLoader(self.dsets['source_aux'],
                                                             batch_size=self.train_hp['test_bs'],
                                                             shuffle=False,
                                                             num_workers=args.num_workers)
                base_network.train(False)
                all_source_features, _, _ = ar_utils.get_features(self.dset_loaders["source_aux"], self.base_network)
                all_target_features, _, _ = ar_utils.get_features(self.dset_loaders["test"], self.base_network)
                self.weights = ar_get_weight.get_weight(
                    all_source_features,
                    all_target_features, 
                    self.loss_hp['rho0'],
                    self.train_hp['seed'],
                    self.train_hp['max_iter_discriminator'],
                    self.train_hp['automatical_adjust'],
                    self.loss_hp['up'],
                    self.loss_hp['low'],
                    i,
                    self.train_hp['multiprocess'],
                    self.loss_hp['c'])
                self.weights = torch.Tensor(self.weights[:])
                self.dset_loaders["source"] = DataLoader(dsets["source"], batch_size=self.train_hp['train_bs'],
                                                    sampler=WeightedRandomSampler(self.weights, num_samples=len(self.weights),
                                                                                  replacement=True),
                                                    num_workers=self.train_hp['num_workers'], drop_last=True)
        if self.train_hp['sampler'] == "uniform_sampler":
            self.train_hp['early_start'] = False
            if self.dset_hp['name'] == "office" and i==200:
                self.train_hp['early_start'] = True
            if i == 0:
                self.weights = torch.ones(len(self.dsets['source_aux']))
            elif i % self.train_hp['weight_update_interval'] == 0 or self.train_hp['early_start']:
                self.base_network.train(False)
                all_source_features, _, _ = ar_utils.get_features(self.dset_loaders['source_aux'], self.base_network)
                all_target_features, _, _ = ar_utils.get_features(self.dset_loaders['test'], self.base_network)
                self.weights = ar_get_weight.get_weight(
                    all_source_features,
                    all_target_features, 
                    self.loss_hp['rho0'],
                    self.train_hp['seed'],
                    self.train_hp['max_iter_discriminator'],
                    self.train_hp['automatical_adjust'],
                    self.loss_hp['up'],
                    self.loss_hp['low'],
                    i,
                    self.train_hp['multiprocess'],
                    self.loss_hp['c'])
                self.weights = torch.Tensor(self.weights[:])
    
    def update_dset_loaders(self, i):
        # Restart iterators if needed
        if i % len(self.dset_loaders["source"]) == 0:
            self.iter_source = iter(self.dset_loaders["source"])
        if i % len(self.dset_loaders["target"]) == 0:
            self.iter_target = iter(self.dset_loaders["target"])

    def save_model(self, name):
        torch.save(self.base_network.state_dict(), os.path.join(self.logger_hp['output_dir'], name))
                         
    def update(self, i):
        """
        Computes the loss and performs one update step.
        """
        self.base_network.train(True)
        xs, ys, ids_source = self.iter_source.next()
        xt, _, _ = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        _, outputs_source = self.base_network(xs)
        features_target, _ = self.base_network(xt)

        ## source (smoothed) cross entropy loss
        if self.loss_hp['label_smooth']:
            if self.train_hp['sampler'] == 'weighted_sampler' or self.train_hp['sampler'] == 'subset_sampler':
                clf_loss = ar_utils.weighted_smooth_cross_entropy(outputs_source, ys)
            else:
                weight = self.weights[ids_source].cuda()
                clf_loss = ar_utils.weighted_smooth_cross_entropy(outputs_source, ys, weight)
        else:
            if self.train_hp['sampler'] == 'weighted_sampler' or self.train_hp['sampler'] == 'subset_sampler':
                clf_loss = ar_utils.weighted_cross_entropy(outputs_source,ys)
            else:
                weight = self.weights[ids_source].cuda()
                clf_loss = ar_utils.weighted_cross_entropy(outputs_source, ys, weight)

        ## target entropy loss
        fc = copy.deepcopy(self.base_network.fc)
        for param in fc.parameters():
            param.requires_grad = False
        softmax_tar_out = torch.nn.Softmax(dim=1)(fc(features_target))
        tar_loss = torch.mean(utils.model_selection.entropy(softmax_tar_out))

        total_loss = clf_loss
        if i>=self.train_hp['start_adapt']:
            adpt_loss = self.loss_hp['ent_weight']*tar_loss
            total_loss = total_loss + adpt_loss
        else:
            adpt_loss= torch.tensor(0.0)
            
        # Updating weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss, clf_loss, adpt_loss
