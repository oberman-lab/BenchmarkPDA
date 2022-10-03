import torch

def get_L2norm_loss_self_driven(x, delta_r):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + delta_r
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return l


