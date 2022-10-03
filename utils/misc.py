import os
from typing import Literal
import torch
import random
import numpy as np

def get_path(dset_hp: dict, split: Literal['train', 'val', 'test', '1shot', '3shot', '25random', '50random', '100random']):
    base_path = os.path.join(dset_hp['root'], dset_hp['name'])
    if 'shot' in split:
        path = os.path.join(base_path, 'image_list_partial_DA_shot', f"{dset_hp['target_domain']}_{split}.txt")
    elif 'random' in split:
        path = os.path.join(base_path, 'image_list_partial_DA_random', f"{dset_hp['target_domain']}_{split}.txt")
    else:
        if dset_hp['use_val']:
            if split == 'train':
                path = os.path.join(base_path, 'image_list_source_split', f"{dset_hp['source_domain']}_train.txt")
            elif split == 'val':
                path = os.path.join(base_path, 'image_list_source_split', f"{dset_hp['source_domain']}_val.txt")
            elif split == 'test':
                path = os.path.join(base_path, 'image_list_partial_DA', f"{dset_hp['target_domain']}_{dset_hp['class_num_target']}_list.txt")
            else:
                raise AttributeError("Not a supported split.")
        else:
            if split == 'train':
                path = os.path.join(base_path, 'image_list_partial_DA', f"{dset_hp['source_domain']}_list.txt")
            elif split == 'test':
                path = os.path.join(base_path, 'image_list_partial_DA', f"{dset_hp['target_domain']}_{dset_hp['class_num_target']}_list.txt")
            else:
                raise AttributeError("Not a supported split when use_val = False.")
    return path


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True