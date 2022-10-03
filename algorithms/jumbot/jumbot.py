import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import utils.optimizers, utils.data_list
import ot
from utils.datasets import transform_train, transform_test
        
from algorithms.base_algorithm import Algorithm

import torch.nn.functional as F

import algorithms.jumbot.utils as jumbot_utils

import warnings

class Jumbot(Algorithm):
    """
    Joint Unbalanced MiniBatch OT (Jumbot)
    """

    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Jumbot, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)

    def set_dset_loaders(self):
        self.dset_loaders = {}
        source_labels = torch.from_numpy(np.array(self.dsets["source"].imgs,)[:,1].astype(int))
        train_batch_sampler = jumbot_utils.BalancedBatchSampler(source_labels, batch_size=self.train_hp['train_bs'])
        self.dset_loaders["source"] = DataLoader(self.dsets["source"],
                                            batch_sampler=train_batch_sampler, 
                                            num_workers=self.train_hp['num_workers'])
        self.dset_loaders["target"] = DataLoader(self.dsets["target"], 
                                                 batch_size=self.train_hp['train_bs'],
                                                 shuffle=True,
                                                 num_workers=self.train_hp['num_workers'],
                                                 drop_last=True)
        self.dset_loaders["test"]   = DataLoader(self.dsets["test"],
                                                 batch_size=self.train_hp['test_bs'],
                                                 shuffle=False,
                                                 num_workers=self.train_hp['num_workers'])

    def update(self, i):
        self.base_network.train(True)
        xs, ys = self.iter_source.next()
        xt, _ = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()

        g_xs, f_g_xs = self.base_network(xs)
        g_xt, f_g_xt = self.base_network(xt)

        pred_xt = F.softmax(f_g_xt, 1)
        clf_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys)
        ys = F.one_hot(ys, num_classes=f_g_xs.shape[1]).float()

        M_embed = torch.cdist(g_xs, g_xt)**2
        M_sce = - torch.mm(ys, torch.transpose(torch.log(pred_xt), 0, 1))
        M = self.loss_hp['eta_1'] * M_embed + self.loss_hp['eta_2'] * M_sce
        a, b = ot.unif(g_xs.size()[0]).astype('float64'), ot.unif(g_xt.size()[0]).astype('float64')
        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.double().detach().cpu().numpy(), reg=self.loss_hp['epsilon'], reg_m=self.loss_hp['tau'])
        pi = torch.from_numpy(pi).float().cuda()        
        
        adpt_loss = self.loss_hp['eta_3'] * torch.sum(pi * M)
        if torch.isnan(adpt_loss):
            warnings.warn(f'adpt_loss is nan at iteration {i}. Set to 0.0 so algorithm can proceed.')
            adpt_loss = torch.tensor(0.0)

        total_loss = clf_loss + adpt_loss 
        # Updating weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, clf_loss, adpt_loss