import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn.functional as F

from utils.datasets import transform_train, transform_test
from utils import network, optimizers, data_list
from utils.model_selection import entropy

import algorithms.etn.utils as etn_utils

from algorithms.base_algorithm import Algorithm

class Etn(Algorithm):
    """
    Etn
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Etn, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_base_network(self):
        self.base_network = network.get_base_network(self.net_hp)
        self.ad_net = network.AdversarialNetwork(self.base_network.output_num(), 1024, self.train_hp['max_iterations'])
        self.classifier_auxiliary = etn_utils.ClassifierAuxiliary(self.base_network.output_num(), self.net_hp['class_num'])

    def prep_for_train(self):
        self.parameter_list = self.base_network.get_parameters() + self.ad_net.get_parameters() + self.classifier_auxiliary.get_parameters()
        self.base_network = self.base_network.cuda()
        self.ad_net = self.ad_net.cuda()
        self.classifier_auxiliary = self.classifier_auxiliary.cuda()
        if self.net_hp['load_net']:
            self.base_network.load_state_dict(torch.load(os.path.join(self.net_hp['load_path'])))
        self.base_network = torch.nn.DataParallel(self.base_network).cuda()
        if self.train_hp['optimizer'] == 'default':
            self.optimizer, self.schedule_param, self.lr_scheduler = optimizers.set_default_optimizer_scheduler(
                self.train_hp, self.parameter_list)

    def update(self, i):
        """
        Computes the loss and performs one update step.
        """
        self.base_network.train(True)
        self.ad_net.train(True)
        xs, ys = self.iter_source.next()
        xt, _ = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()


        g_xs, f_g_xs = self.base_network(xs)
        g_xt, f_g_xt = self.base_network(xt)
        
        predict_prob_source = F.softmax(f_g_xs, 1)
        predict_prob_target = F.softmax(f_g_xt, 1)
        
        
#         # =========================forward pass
#         im_source = im_source.to(output_device)
#         im_target = im_target.to(output_device)

#         fc1_s = feature_extractor.forward(im_source)
#         fc1_t = feature_extractor.forward(im_target)

#         fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
#         fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = self.ad_net(g_xs)
        domain_prob_discriminator_target = self.ad_net(g_xt)

        predict_prob_source_aug, domain_prob_source_aug = self.classifier_auxiliary.forward(g_xs.detach())
        predict_prob_target_aug, domain_prob_target_aug = self.classifier_auxiliary.forward(g_xt.detach())

        # ============================== compute loss
        weight = (1.0 - domain_prob_source_aug)
        weight = weight / (torch.mean(weight, dim=0, keepdim=True) + 1e-10)
        weight = weight.detach()

        # ============================== cross entropy loss, it receives logits as its inputs
        clf_loss = nn.CrossEntropyLoss(reduction='none')(f_g_xs, ys).view(-1, 1)
        clf_loss = torch.mean(clf_loss * weight, dim=0, keepdim=True).squeeze()
#         print(f'clf_loss = {clf_loss}')
        tmp = weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source+1e-10, torch.ones_like(domain_prob_discriminator_source))
        adv_loss = torch.mean(tmp, dim=0).squeeze()
        adv_loss += nn.BCELoss()(domain_prob_discriminator_target+1e-10, torch.zeros_like(domain_prob_discriminator_target))
#         print(f'adv_loss = {adv_loss}')
        ce_aug = nn.BCELoss(reduction='none')(predict_prob_source_aug, F.one_hot(ys, self.net_hp['class_num']).float())
        ce_aug = torch.sum(ce_aug) / ys.numel()
#         print(f'ce_aug = {ce_aug}')
        adv_loss_aug = nn.BCELoss()(domain_prob_source_aug, torch.ones_like(domain_prob_source_aug))
        adv_loss_aug += nn.BCELoss()(domain_prob_target_aug, torch.zeros_like(domain_prob_target_aug))
#         print(f'adv_loss_aug = {adv_loss_aug}')
        entropy = etn_utils.EntropyLoss(predict_prob_target)
#         print(f'entropy = {entropy}')
        adpt_loss = self.loss_hp['adv_loss_tradeoff'] * adv_loss + self.loss_hp['entropy_tradeoff'] * entropy + \
                   self.loss_hp['adv_loss_aug_tradeoff'] * adv_loss_aug + self.loss_hp['ce_aug_tradeoff'] * ce_aug
        adpt_loss = adpt_loss.squeeze()

        total_loss = clf_loss + adpt_loss

#         print(f'i = {i} Loss: {total_loss:.2f} = {clf_loss:.2f} + {adpt_loss:.2f}')
        # Updating weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss, clf_loss, adpt_loss