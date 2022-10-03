import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import transform_train, transform_test
from utils import network, optimizers, data_list
from utils.model_selection import entropy

import algorithms.ba3us.utils as ba3us_utils

from algorithms.base_algorithm import Algorithm

class BA3US(Algorithm):
    """
    BA3US: Balanced Adversarial Alignment (BAA) and Adaptive Uncertainty Suppression (AUS)
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(BA3US, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_base_network(self):
        self.base_network = network.get_base_network(self.net_hp)
        self.ad_net = network.AdversarialNetwork(self.base_network.output_num(), 1024, self.train_hp['max_iterations'])

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
        self.class_weight = None
        self.total_epochs = self.train_hp['max_iterations'] // self.train_hp['test_interval']

    def update_dsets(self, i):
        if i % self.train_hp['test_interval'] == 0:
            if self.loss_hp['mu'] > 0:
                epoch = i // self.train_hp['test_interval']
                self.len_share = int(max(0, (self.train_hp['train_bs'] // self.loss_hp['mu']) * (1 - epoch / self.total_epochs)))
            elif self.loss_hp['mu'] == 0:
                self.len_share = 0  # no augmentation
            else:
                self.len_share = int(self.train_hp['train_bs'] // abs(self.loss_hp['mu']))  # fixed augmentation

            self.dset_loaders['middle'] = None
            if not self.len_share == 0:
                self.dset_loaders['middle'] = DataLoader(self.dsets['source'], 
                                                         batch_size=self.len_share, 
                                                         shuffle=True,
                                                         num_workers=self.train_hp['num_workers'],
                                                         drop_last=True)
                self.iter_middle = iter(self.dset_loaders['middle'])

    def update_dset_loaders(self, i):
        # Restart iterators if needed
        if i % len(self.dset_loaders["source"]) == 0:
            self.iter_source = iter(self.dset_loaders["source"])
        if i % len(self.dset_loaders["target"]) == 0:
            self.iter_target = iter(self.dset_loaders["target"])
        if self.dset_loaders['middle'] is not None and i % len(self.dset_loaders['middle']) == 0:
            self.iter_middle = iter(self.dset_loaders['middle'])

    def update(self, i):
        """
        Computes the loss and performs one update step.
        """
        self.base_network.train(True)
        self.ad_net.train(True)
        xs, ys = self.iter_source.next()
        xt, _ = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()
        if self.class_weight is not None and self.loss_hp['weight_cls'] and self.class_weight[ys].sum() == 0:
            total_loss = torch.tensor(0.0)
            src_loss = torch.tensor(0.0)
        else:
            features_source, outputs_source = self.base_network(xs)
            features_target, outputs_target = self.base_network(xt)

            if self.dset_loaders['middle'] is not None:
                inputs_middle, labels_middle = self.iter_middle.next()
                features_middle, outputs_middle = self.base_network(inputs_middle.cuda())
                features = torch.cat((features_source, features_target, features_middle), dim=0)
                outputs = torch.cat((outputs_source, outputs_target, outputs_middle), dim=0)
            else:
                features = torch.cat((features_source, features_target), dim=0)
                outputs = torch.cat((outputs_source, outputs_target), dim=0)

            cls_weight = torch.ones(outputs.size(0)).cuda()
            if self.class_weight is not None and self.loss_hp['weight_aug']:
                cls_weight[0:self.train_hp['train_bs']] = self.class_weight[ys]
                if self.dset_loaders['middle'] is not None:
                    cls_weight[2*self.train_hp['train_bs']::] = self.class_weight[labels_middle]

            # compute source cross-entropy loss
            if self.class_weight is not None and self.loss_hp['weight_cls']:
                src_ = torch.nn.CrossEntropyLoss(reduction='none')(outputs_source, ys)
                weight = self.class_weight[ys].detach()
                src_loss = torch.sum(weight * src_) / (1e-8 + torch.sum(weight).item())
            else:
                src_loss = torch.nn.CrossEntropyLoss()(outputs_source, ys)

            softmax_out = torch.nn.Softmax(dim=1)(outputs)
            transfer_loss = ba3us_utils.DANN(features, self.ad_net, entropy(softmax_out), network.calc_coeff(i, 1, 0, 10, self.train_hp['max_iterations']), cls_weight, self.len_share)       

            softmax_tar_out = torch.nn.Softmax(dim=1)(outputs_target)
            tar_loss = torch.mean(entropy(softmax_tar_out))

            total_loss = src_loss + transfer_loss + self.loss_hp['ent_weight'] * tar_loss
            if self.loss_hp['cot_weight'] > 0:
                if self.class_weight is not None and self.loss_hp['weight_cls']:
                    cot_loss = ba3us_utils.marginloss(outputs_source, ys, self.net_hp['class_num'], alpha=self.loss_hp['alpha'], weight=self.class_weight[ys].detach())
                else:
                    cot_loss = ba3us_utils.marginloss(outputs_source, ys, self.net_hp['class_num'], alpha=self.loss_hp['alpha'])
                total_loss += cot_loss * self.loss_hp['cot_weight']

#             print(f'i = {i} Loss: {total_loss:.2f} = {src_loss:.2f} + {total_loss-src_loss:.2f}')

            # Updating weights
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            return total_loss, src_loss, total_loss-src_loss