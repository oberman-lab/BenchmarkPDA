import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import utils.optimizers, utils.data_list
from utils.datasets import transform_train, transform_test

from algorithms.jumbot.jumbot import Jumbot
import ot

def SCE(proba1, proba2, eta_4=0.01):
    '''
        Compute the symmetric cross entropy
        args : 
        - proba1 : probability vector batches (one hot vectors)
        - proba2 : probability vector batches (one hot vectors)
        Return :
        Symmetric cross entropy
        
    '''

    proba1_clamp = torch.clamp(proba1, 1e-7, 1)
    proba2_clamp = torch.clamp(proba2, 1e-7, 1)
    sce = - eta_4 * torch.mm(proba1, torch.transpose(torch.log(proba2_clamp), 0, 1))
    sce += - torch.mm(torch.log(proba1_clamp), torch.transpose(proba2, 0, 1))
    return sce


class Mixunbot(Jumbot):
    """
    Mixup Unbalanced Optimal Optimal Transport (Mixunbot)
    """
    def __init__(self, dset_hp, loss_hp, train_hp, net_hp, logger_hp):
        super(Mixunbot, self).__init__(dset_hp, loss_hp, train_hp, net_hp, logger_hp)
        
    def update(self, i):
        self.base_network.train(True)
        xs, ys = self.iter_source.next()
        xt, _ = self.iter_target.next()
        xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()
        
        ### MixUp
        l = np.random.beta(self.loss_hp['beta'], self.loss_hp['beta'])
        idxs = torch.randperm(xs.size(0))
        idxt = torch.randperm(xt.size(0))

        xs = l * xs + (1 - l) * xs[idxs]
        xt = l * xt + (1 - l) * xt[idxt]

        g_xs, f_g_xs = self.base_network(xs)
        g_xt, f_g_xt = self.base_network(xt)

        pred_xt = F.softmax(f_g_xt, 1)
        
        clf_loss = l*torch.nn.CrossEntropyLoss()(f_g_xs, ys) + (1-l)*torch.nn.CrossEntropyLoss()(f_g_xs, ys[idxs])        
        ys = F.one_hot(ys, num_classes=f_g_xs.shape[1]).float()
        tilde_ys = l * ys + (1-l) * ys[idxs] 
        
        M_embed = torch.cdist(g_xs, g_xt)**2
        M_sce = SCE(tilde_ys, pred_xt, eta_4=self.loss_hp['eta_4'])  # distance between labels
        M = self.loss_hp['eta_1'] * M_embed + self.loss_hp['eta_2'] * M_sce
        a, b = ot.unif(g_xs.size()[0]).astype('float64'), ot.unif(g_xt.size()[0]).astype('float64')
        pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(), reg=self.loss_hp['epsilon'], reg_m=self.loss_hp['tau'])
        pi = torch.from_numpy(pi).float().cuda()
        adpt_loss = self.loss_hp['eta_3'] * torch.sum(pi * M)

        total_loss = clf_loss + adpt_loss 

        # Updating weights
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss, clf_loss, adpt_loss