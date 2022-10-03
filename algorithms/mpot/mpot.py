import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import utils.optimizers, utils.data_list
import ot
from utils.datasets import transform_train, transform_test

from algorithms.jumbot.jumbot import Jumbot

import torch.nn.functional as F

import algorithms.jumbot.utils as jumbot_utils

import warnings

class Mpot(Jumbot):
    """
    Mini-Batch Partial OT (mPOT)
    """

    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Mpot, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)


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
#         a, b = ot.unif(g_xs.size()[0]).astype('float64'), ot.unif(g_xt.size()[0]).astype('float64')
        a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
        
        M_cpu = M.detach().cpu().numpy()
#         M_cpu = M.double().detach().cpu().numpy()
        
        if i <= (self.train_hp['max_iterations'] / 2):
            adap_mass = (self.loss_hp['mass']-0.01) / (self.train_hp['max_iterations']/2) * i + 0.01
        else:
            adap_mass = self.loss_hp['mass']
        if self.loss_hp['epsilon'] == 0:
            pi = ot.partial.partial_wasserstein(a, b, M_cpu, adap_mass)
        else:
            pi = ot.partial.entropic_partial_wasserstein(a, b, M_cpu, m=adap_mass, reg=self.loss_hp['epsilon'])        
        
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