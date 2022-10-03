import torch

def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer

schedule_dict = {"inv":inv_lr_scheduler}

def set_default_optimizer_scheduler(train_hparams, parameter_list):
    optimizer_config = {"type":torch.optim.SGD,
                        "optim_params": {
                            'lr':train_hparams['lr'],
                            "momentum":train_hparams['momentum'],
                            "weight_decay":train_hparams['weight_decay'],
                            "nesterov":train_hparams['nesterov']}, 
                        "lr_type":"inv",
                        "lr_param":{
                            "lr":train_hparams['lr'],
                            "gamma":train_hparams['gamma'],
                            "power":train_hparams['power']}}
    optimizer = optimizer_config["type"](parameter_list,**(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = schedule_dict[optimizer_config["lr_type"]]
    return optimizer, schedule_param, lr_scheduler
