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
                    default='logs_hp_search_nonlinear',
                    help="Path to folder to save the logs")
parser.add_argument('--dset', type=str, 
                    default='office-home',
                    help="Choice of dataset",
                    choices=['office-home', 'visda', 'domainnet'])
parser.add_argument('--method', type=str,
                    default='ar',
                    help="Choice of partial domain adaptation method",
                    choices=algorithms_dict.keys())
parser.add_argument('--net', type=str, 
                    default='ResNet50',
                    help="Choice of neural network architecture",
                    choices=net_dict.keys())
parser.add_argument('--source_domain', type=str, 
                    default='Art',
                    help="Choice of source domain")
parser.add_argument('--target_domain', type=str, 
                    default='Clipart',
                    help="Choice of source domain")
parser.add_argument('--seed', type=int, 
                    default=2020,
                    help="Choice of seed")

args = parser.parse_args()

dset_hp, domains = get_dset_hp(args.dset, args.data_folder)
dset_hp['use_val'] = True

net_hp = get_net_hp_nonlinear(dset_hp, args.net)

logger_hp = get_logger_hp_search(args.dset)


if args.source_domain in domains:
    dset_hp['source_domain'] = args.source_domain
else:
    raise ValueError('Not an available domain for the dataset chosen.')
if args.target_domain in domains:
    dset_hp['target_domain'] = args.target_domain
else:
    raise ValueError('Not an available domain for the dataset chosen.')
if dset_hp['source_domain'] == dset_hp['target_domain']:
    raise ValueError('Source and target domains should be different.')

loss_hp = get_loss_hp_default(args.method, dset_hp)
dset_hp = dset_hp_update_paths_task(dset_hp, logger_hp)
search_space = get_search_space(args.method)

train_hp = get_train_hp_search(args.method, dset_hp)
train_hp['seed'] = args.seed
for hp_params in itertools.product(*[iter(search_space[key]) for key in search_space.keys()]):
    if args.method == 'ar':
        if hp_params[1] != -hp_params[2]: # restricts up == -low
            continue
    for ix, key in enumerate(search_space.keys()):
        loss_hp[key] = hp_params[ix]

    # Set randoms seed for reproducibility
    set_seeds(train_hp['seed'])

    # Find output_dir
    output_dir = os.path.join(args.logs_folder, args.method, net_hp['net'], dset_hp['name'], dset_hp['task'])     
    for ix, key in enumerate(search_space.keys()):
        output_dir = os.path.join(output_dir, f'{key}_{hp_params[ix]}')
    logger_hp['output_dir'] = os.path.join(output_dir, f"seed_{train_hp['seed']}", 'run_0')
    
    # Train with specified hyper-parameters
    print(os.path.exists(logger_hp['output_dir']))
    if not os.path.exists(logger_hp['output_dir']):
        os.makedirs(logger_hp['output_dir'], exist_ok=True)
        algorithm = algorithms_dict[args.method](dset_hp, loss_hp, train_hp, net_hp, logger_hp)
        train(algorithm)