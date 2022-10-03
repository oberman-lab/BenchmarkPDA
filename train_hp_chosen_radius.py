import os, random, argparse
import itertools
import numpy as np
import torch
from train import train # Training script
from algorithms import algorithms_dict # Dictionary of methods available
from utils.network import net_dict # Dictionary of pre-trained models available
from utils.hp_functions import * # Helper functions to load default hyper-parameters
from utils.misc import *

parser = argparse.ArgumentParser(description='Partial Domain Adaptation')
parser.add_argument('--data_folder', type=str,
                    default='datasets',
                    help="Path to datatasets")
parser.add_argument('--logs_folder', type=str,
                    default='logs_hp_chosen_radius',
                    help="Path to folder to save the logs")
parser.add_argument('--dset', type=str, 
                    default='office-home',
                    help="Choice of dataset",
                    choices=['office-home', 'visda', 'domainnet'])
parser.add_argument('--net', type=str, 
                    default='ResNet50',
                    help="Choice of neural network architecture",
                    choices=net_dict.keys())
parser.add_argument('--seed', type=int, 
                    default=2020,
                    help="Choice of seed")

args = parser.parse_args()

dset_hp, domains = get_dset_hp(args.dset, args.data_folder)
dset_hp['use_val'] = True
net_hp = get_net_hp_default(dset_hp, args.net)
logger_hp = get_logger_hp_chosen(args.dset)

hp = np.load('results/hp_chosen_radius.npy', allow_pickle=True).item()
hp = hp[args.dset]

for method in ['ar']:
    for metric in ['oracle']:
        for seed in range(2020,2023):
            for dset_hp['source_domain'] in domains:
                for dset_hp['target_domain'] in domains:
                    if dset_hp['source_domain'] != dset_hp['target_domain']:                    
                        loss_hp = get_loss_hp_default(method, dset_hp)
                        dset_hp = dset_hp_update_paths_task(dset_hp, logger_hp)
                        train_hp = get_train_hp_chosen(method, dset_hp)
                        train_hp['seed'] = seed
                        for key in hp[method][metric]:
                            loss_hp[key] = hp[method][metric][key]
                        set_seeds(train_hp['seed'])
                        # Find output_dir
                        output_dir = os.path.join(args.logs_folder, method, net_hp['net'], dset_hp['name'], dset_hp['task'])     
                        for key in hp[method][metric]:
                            output_dir = os.path.join(output_dir, f'{key}_{hp[method][metric][key]}')
                        logger_hp['output_dir'] = os.path.join(output_dir, f"seed_{train_hp['seed']}", 'run_0')
                        # Train with specified hyper-parameters
                        if not os.path.exists(logger_hp['output_dir']):
                            os.makedirs(logger_hp['output_dir'], exist_ok=True)
                            algorithm = algorithms_dict[method](dset_hp, loss_hp, train_hp, net_hp, logger_hp)
                            train(algorithm)