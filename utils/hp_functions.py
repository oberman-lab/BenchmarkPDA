from utils.misc import get_path

# Dataset Hyper-Parameters
def get_dset_hp(dset, data_folder):
    dset_hp = {'name': dset, 'root': data_folder}
    if dset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
        dset_hp['class_num'] = 65
        dset_hp['class_num_target'] = 25
    elif dset == 'visda':
        domains = ['train', 'validation']
        dset_hp['class_num'] = 12
        dset_hp['class_num_target'] = 6
    elif dset == 'domainnet':
        domains = ['clipart', 'painting', 'real', 'sketch']
        dset_hp['class_num'] = 126
        dset_hp['class_num_target'] = 40
    return dset_hp, domains

# Logger Hyper-Parameters
def get_logger_hp_default():
    logger_hp = {
        'save_model_source': False,
        'save_model_dev': False,
        'save_model_ent': False,
        'save_model_oracle': False,
        'save_model_final': True,
        'save_outputs_evl':False,
        'filename': 'log',
        'printQ': True
    }
    return logger_hp

def get_logger_hp_source_only():
    logger_hp = {
        'model_selection_metrics': ['t_acc'],
        'save_models': ['t_acc'],
        'save_model_final': False,
        'save_outputs_evl':False,
        'filename': 'log',
        'printQ': True
    }
    return logger_hp

def get_logger_hp_search(dset):
    logger_hp = {
        'model_selection_metrics': ['s_acc', 't_acc', 'ent', 'snd', 'dev_lr', 'dev_mlp', 'dev_svm',
                                    '1shot_acc', '1shot_10crop_acc', '3shot_acc', '3shot_10crop_acc'],
        'save_models': [],
        'save_model_final': True,
        'save_outputs_evl':False,
        'filename': 'log',
        'printQ': True
    }
    if dset == 'office-home':
        logger_hp['model_selection_metrics'] = ['s_acc', 't_acc', 'ent', 'snd', 'dev_svm',
                                                '25random_acc', '50random_acc', '100random_acc',
                                                '100random_10crop_acc', '1shot_acc', '1shot_10crop_acc', '3shot_acc',
                                                '3shot_10crop_acc']
    elif dset == 'visda':
        logger_hp['model_selection_metrics'] = ['s_acc', 't_acc', 'ent', 'snd', 'dev_svm',
                                                '25random_acc', '50random_acc', '100random_acc',
                                                '100random_10crop_acc', '1shot_acc', '1shot_10crop_acc', '3shot_acc',
                                                '3shot_10crop_acc']
    elif dset == 'domainnet':
        logger_hp['model_selection_metrics'] = ['s_acc', 't_acc', 'ent', 'snd', 'dev_svm',
                                                '25random_acc', '50random_acc', '100random_acc',
                                                '100random_10crop_acc', '1shot_acc', '1shot_10crop_acc', '3shot_acc',
                                                '3shot_10crop_acc']
    return logger_hp

def get_logger_hp_chosen(dset):
    logger_hp = {
        'model_selection_metrics': ['s_acc', 't_acc', 'ent', 'snd', 'dev_svm','25random_acc', '50random_acc', '100random_acc',
                                    '100random_10crop_acc', '1shot_acc', '1shot_10crop_acc', '3shot_acc', '3shot_10crop_acc'],
        'save_models': [],
        'save_model_final': True,
        'save_outputs_evl':False,
        'filename': 'log',
        'printQ': True
    }
    return logger_hp


# Network Hyper-Parameters
def get_net_hp_default(dset_hp, name):
    net_hp = {'net':name, 'use_bottleneck':True, 'bottleneck_dim':256, 'class_num':dset_hp['class_num'], 'radius':0.0, 'use_slr':False, 'load_net':False, 'nonlinear':False}
    return net_hp

def get_net_hp_nonlinear(dset_hp, name):
    net_hp = {'net':name, 'class_num':dset_hp['class_num'], 'load_net':False, 'nonlinear':True}
    return net_hp


from algorithms.ar.utils import recommended_bottleneck_dim
# Network Hyper-Parameters used in AR paper (without SLR head)
def get_net_hp_default_ar_linear(dset_hp):
    net_hp = get_net_hp_default(dset_hp)
    if dset_hp['name'] == 'office-home':
        net_hp['radius'] = 10.0
    elif dset_hp['name'] == 'visda':
        net_hp['radius'] = 5.0
    elif dset_hp['name'] == 'domainnet':
        net_hp['radius'] = 20.0
    net_hp['bottleneck_dim'] = recommended_bottleneck_dim(dset_hp['class_num'])
    return net_hp

# Loss Hyper-Parameters
def get_loss_hp_original(method, dset_hp):
    if method == 'jumbot':
        if dset_hp['name'] == 'office-home':
            loss_hp = {'name':'jumbot', 'epsilon':0.01, 'tau':0.06, 'eta_1':0.003, 'eta_2':0.75, 'eta_3':10.0}
    return loss_hp

def get_loss_hp_default(method, dset_hp):
    if method == 'jumbot':
        if dset_hp['name'] == 'office-home':
            loss_hp = {'name':'jumbot', 'epsilon':0.01, 'tau':0.005, 'eta_1':0.0001, 'eta_2':1.0, 'eta_3':10.0}
        elif dset_hp['name'] == 'visda':
            loss_hp = {'name':'jumbot', 'epsilon':0.01, 'tau':0.01, 'eta_1':0.001, 'eta_2':1.0, 'eta_3':5.0}
        elif dset_hp['name'] == 'domainnet':
            loss_hp = {'name':'jumbot', 'epsilon':0.01, 'tau':0.01, 'eta_1':0.0001, 'eta_2':0.5, 'eta_3':20.0}
    elif method == 'deepjdot':
        if dset_hp['name'] == 'office-home':
            loss_hp = {'name':'deepjdot', 'eta_1':0.001, 'eta_2':0.0001, 'eta_3':1.0}
        else:
            raise NotImplementedError
    elif method == 'etn':
        if dset_hp['name'] == 'office-home':
            loss_hp = {'name':'etn', 'adv_loss_tradeoff': 1.0, 'entropy_tradeoff': 0.2,'adv_loss_aug_tradeoff': 10.0, 'ce_aug_tradeoff': 1.0}
        else:
            raise NotImplementedError
    elif method == 'mixunbot':
        loss_hp = {'name':'mixunbot', 'epsilon':0.01, 'tau':0.05, 'eta_1':0.001, 'eta_2':0.05, 'eta_3':15.0, 'eta_4': 0.01, 'beta': 0.2}
    elif method == 'mpot':
        loss_hp = {'name':'mpot', 'epsilon':0.5, 'eta_1':0.03, 'eta_2':7.5, 'eta_3':1.0, 'mass': 0.325}
    elif method == 'safn':
        loss_hp = {'name':'safn', 'lambda':0.05, 'delta_r': 1.0}
    elif method == 'pada':
        loss_hp = {'name':'pada', 'lambda':1.0}
    elif method == 'source_only_plus':
        loss_hp = {'name':'source_only_plus'}
    elif method == 'ar':
        loss_hp = {'name':'ar', 'label_smooth':False, 'rho0':5.0, 'up':5.0, 'low':-5.0, 'c':1.2, 'ent_weight':0.1}
        if (dset_hp['name'] == 'visda') and (dset_hp['source_domain'] == 'validation'):
            loss_hp['low'] = 1.0        
    elif method == 'ba3us':
        loss_hp = {'name':'ba3us', 'mu':4, 'ent_weight':0.1, 'cot_weight':1.0, 'weight_aug':True, 'weight_cls':True, 'alpha':1.0}
    return loss_hp


def get_loss_hp_chosen(method, dset_hp):
    if method == 'jumbot':
        loss_hp = {'name':'jumbot', 'epsilon':0.01}
    elif method == 'etn':
        loss_hp = {'name':'etn'}
    elif method == 'mixunbot':
        loss_hp = {'name':'mixunbot', 'epsilon':0.01, 'beta': 0.2}
    elif method == 'mpot':
        loss_hp = {'name':'mpot', 'epsilon':0.5}
    elif method == 'safn':
        loss_hp = {'name':'safn'}
    elif method == 'pada':
        loss_hp = {'name':'pada'}
    elif method == 'ar':
        loss_hp = {'name':'ar', 'label_smooth':False, 'c':1.2}
    elif method == 'ba3us':
        loss_hp = {'name':'ba3us', 'mu':4, 'weight_aug':True, 'weight_cls':True, 'alpha':1.0}
    elif method == 'source_only_plus':
        loss_hp = {'name':'source_only_plus'}
    return loss_hp

# Train Hyper-Parameters
def get_train_hp_default(method, dset_hp):
    train_hp = {
        'optimizer': 'default',
        'lr':1e-3,
        'momentum':0.9,
        'gamma':0.001,
        'power':0.75,
        'weight_decay':5e-4,
        'nesterov':True,
        'num_workers':4,
        'train_bs':36,
        'test_bs':72,
    }
    if 'source_only' == method:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 1000
            train_hp['test_interval'] = 50
        else:
            raise NotImplementedError
    elif 'source_only_plus' == method:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 1000
            train_hp['test_interval'] = 100
        else:
            raise NotImplementedError
    else:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 5000
            train_hp['test_interval'] = 500
        elif dset_hp['name'] == 'visda':
            train_hp['max_iterations'] = 40000
            train_hp['test_interval'] = 1000
        elif dset_hp['name'] == 'domainnet':
            train_hp['max_iterations'] = 20000
            train_hp['test_interval'] = 1000

    if (method == 'jumbot') or (method == 'deepjdot') or (method == 'mixunbot') or (method == 'mpot'):
        if dset_hp['name'] == 'office-home':
            train_hp['train_bs'] = 65
            train_hp['test_bs'] = 130
        elif dset_hp['name'] == 'domainnet':
            train_hp['train_bs'] = 126
            train_hp['test_bs'] = 126
    elif method == 'ar':
        train_hp['automatical_adjust'] = True
        if dset_hp['name'] == 'office-home':
            train_hp['start_adapt'] = 0
            train_hp['max_iter_discriminator'] = 3000
            train_hp['multiprocess'] = False
            train_hp['weight_update_interval'] = 500
            train_hp['sampler'] = "uniform_sampler"
        elif dset_hp['name'] == 'visda':
            train_hp['start_adapt'] = 0
            train_hp['max_iter_discriminator'] = 6000
            if dset_hp['source_domain'] == 'train':
                train_hp['multiprocess'] = False
                train_hp['weight_update_interval'] = 3000
                train_hp['sampler'] = "weighted_sampler"
            elif dset_hp['source_domain'] == 'validation':
                train_hp['multiprocess'] = True
                train_hp['weight_update_interval'] = 1000
                train_hp['sampler'] = "uniform_sampler"
        elif dset_hp['name'] == 'domainnet':
            train_hp['start_adapt'] = 1000
            train_hp['max_iter_discriminator'] = 6000
            train_hp['multiprocess'] = False
            train_hp['weight_update_interval'] = 1000
            train_hp['sampler'] = "weighted_sampler"
    elif method == 'pada':
        train_hp['weight_update_interval'] = 500
    return train_hp

# Train Hyper-Parameters
def get_train_hp_source_only(dset_hp):
    train_hp = {
        'optimizer': 'default',
        'lr':1e-3,
        'momentum':0.9,
        'gamma':0.001,
        'power':0.75,
        'weight_decay':5e-4,
        'nesterov':True,
        'num_workers':4,
        'train_bs':36,
        'test_bs':72,
    }
    if 'source_only' == method:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 1000
            train_hp['test_interval'] = 50
        else:
            raise NotImplementedError
    elif 'source_only_plus' == method:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 1000
            train_hp['test_interval'] = 100
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return train_hp

def get_train_hp_search(method, dset_hp):
    train_hp = {
        'optimizer': 'default',
        'lr':1e-3,
        'momentum':0.9,
        'gamma':0.001,
        'power':0.75,
        'weight_decay':5e-4,
        'nesterov':True,
        'num_workers':4,
        'train_bs':36,
        'test_bs':72,
    }
    if 'source_only' == method:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 1000
            train_hp['test_interval'] = 50
        else:
            raise NotImplementedError
    elif 'source_only_plus' == method:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 1000
            train_hp['test_interval'] = 100
        elif dset_hp['name'] == 'visda':
            train_hp['max_iterations'] = 5000
            train_hp['test_interval'] = 500
        else:
            raise NotImplementedError
    else:
        if dset_hp['name'] == 'office-home':
            train_hp['max_iterations'] = 5000
            train_hp['test_interval'] = 500
        elif dset_hp['name'] == 'visda':
            train_hp['max_iterations'] = 10000
            train_hp['test_interval'] = 1000
        elif dset_hp['name'] == 'domainnet':
            train_hp['max_iterations'] = 20000
            train_hp['test_interval'] = 1000

    if (method == 'jumbot') or (method == 'deepjdot') or (method == 'mixunbot') or (method == 'mpot'):
        if dset_hp['name'] == 'office-home':
            train_hp['train_bs'] = 65
            train_hp['test_bs'] = 130
        elif dset_hp['name'] == 'domainnet':
            train_hp['train_bs'] = 126
            train_hp['test_bs'] = 126
    elif method == 'ar':
        train_hp['automatical_adjust'] = True
        train_hp['multiprocess'] = False
        train_hp['sampler'] = "weighted_sampler"
        train_hp['start_adapt'] = int(train_hp['max_iterations']*0.05)
        train_hp['weight_update_interval'] = train_hp['test_interval']
        if dset_hp['name'] == 'office-home':
            train_hp['max_iter_discriminator'] = 3000
        elif dset_hp['name'] == 'visda':
            train_hp['max_iter_discriminator'] = 6000
        elif dset_hp['name'] == 'domainnet':
            train_hp['max_iter_discriminator'] = 6000
    elif method == 'pada':
        train_hp['weight_update_interval'] = 500
    return train_hp

def get_train_hp_chosen(method, dset_hp):
    return get_train_hp_search(method, dset_hp)


def dset_hp_update_paths_task(dset_hp, logger_hp):
    dset_hp['s_dset_path'] = get_path(dset_hp, 'train')
    if dset_hp['use_val']:
        dset_hp['v_dset_path'] = get_path(dset_hp, 'val')
    if ('1shot_acc' in logger_hp['model_selection_metrics']) or ('1shot_10crop_acc' in logger_hp['model_selection_metrics']):
        dset_hp['t_1shot_dset_path'] = get_path(dset_hp, '1shot')
    if ('3shot_acc' in logger_hp['model_selection_metrics']) or ('3shot_10crop_acc' in logger_hp['model_selection_metrics']):
        dset_hp['t_3shot_dset_path'] = get_path(dset_hp, '3shot')
    if ('25random_acc' in logger_hp['model_selection_metrics']) or ('25random_10crop_acc' in logger_hp['model_selection_metrics']):
        dset_hp['t_25random_dset_path'] = get_path(dset_hp, '25random')
    if ('50random_acc' in logger_hp['model_selection_metrics']) or ('50random_10crop_acc' in logger_hp['model_selection_metrics']):
        dset_hp['t_50random_dset_path'] = get_path(dset_hp, '50random')
    if ('100random_acc' in logger_hp['model_selection_metrics']) or ('100random_10crop_acc' in logger_hp['model_selection_metrics']):
        dset_hp['t_100random_dset_path'] = get_path(dset_hp, '100random')
    dset_hp['t_dset_path'] = get_path(dset_hp, 'test')
    dset_hp['task'] = dset_hp['source_domain'][0].upper() + dset_hp['target_domain'][0].upper()
    return dset_hp


def get_search_space(method):
    if (method == 'jumbot') or (method == 'mixunbot'):
        search_space = {'tau': [0.001, 0.01, 0.1],
                        'eta_1': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                        'eta_2': [0.1, 0.5, 1.],
                        'eta_3': [5, 10, 20]}
    elif method == 'ar':
        search_space = {'rho0': [2.5, 5.0, 7.5, 10.0],
                        'up':[1.0, 5.0, 10.0],
                        'low':[-10.0, -5.0, -1.0],
                        'ent_weight': [0.01, 0.1, 1.0]}
    elif method == 'ba3us':
        search_space = {'cot_weight': [0.1, 0.5, 1, 5, 10], 'ent_weight': [0.01, 0.05, 0.1, 0.5, 1]}
    elif method == 'mpot':
        search_space = {'epsilon': [0.5, 1.0, 1.5],
                        'eta_1': [0.0001, 0.001, 0.01, 0.1, 1.0],
                        'eta_2': [0.1, 1.0, 5.0, 10.0],
                        'mass': [0.1, 0.2, 0.3, 0.4]}
    elif method == 'safn':
        search_space = {'lambda': [0.005, 0.01, 0.05, 0.1, 0.5], 'delta_r': [0.01, 0.1, 1.0]}
    elif method == 'pada':
        search_space = {'lambda': [0.1, 0.5, 1.0, 5.0, 10.0]}
    elif method == 'etn':
        search_space = {'adv_loss_tradeoff': [0.1, 0.5, 1.0, 5.0], # lambda_1
                        'entropy_tradeoff': [0.05, 0.1, 0.2, 0.5], # lambda_2
                        'adv_loss_aug_tradeoff': [0.5, 1.0, 5.0, 10.0], # lambda_3
                        'ce_aug_tradeoff': [0.1, 0.5, 1.0, 5.0], # lambda_4
                       }
    elif method == 'source_only_plus':
        search_space = {}
    return search_space


def get_search_space(method):
    if (method == 'jumbot') or (method == 'mixunbot'):
        search_space = {'tau': [0.001, 0.01, 0.1],
                        'eta_1': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                        'eta_2': [0.1, 0.5, 1.],
                        'eta_3': [5, 10, 20]}
    elif method == 'ar':
        search_space = {'rho0': [2.5, 5.0, 7.5, 10.0],
                        'up':[5.0, 10.0],
                        'low':[-10.0, -5.0],
                        'ent_weight': [0.01, 0.1, 1.0]}
    elif method == 'ba3us':
        search_space = {'cot_weight': [0.1, 0.5, 1, 5, 10], 'ent_weight': [0.01, 0.05, 0.1, 0.5, 1]}
    elif method == 'mpot':
        search_space = {'epsilon': [0.5, 1.0, 1.5],
                        'eta_1': [0.0001, 0.001, 0.01, 0.1, 1.0],
                        'eta_2': [0.1, 1.0, 5.0, 10.0],
                        'mass': [0.1, 0.2, 0.3, 0.4]}
    elif method == 'safn':
        search_space = {'lambda': [0.005, 0.01, 0.05, 0.1, 0.5], 'delta_r': [0.01, 0.1, 1.0]}
    elif method == 'pada':
        search_space = {'lambda': [0.1, 0.5, 1.0, 5.0, 10.0]}
    elif method == 'etn':
        search_space = {'adv_loss_tradeoff': [0.1, 0.5, 1.0, 5.0], # lambda_1
                        'entropy_tradeoff': [0.05, 0.1, 0.2, 0.5], # lambda_2
                        'adv_loss_aug_tradeoff': [0.5, 1.0, 5.0, 10.0], # lambda_3
                        'ce_aug_tradeoff': [0.1, 0.5, 1.0, 5.0], # lambda_4
                       }
    elif method == 'source_only_plus':
        search_space = {}
    return search_space

def get_search_space_with_radius(method):
    if (method == 'jumbot') or (method == 'mixunbot'):
        search_space = {'tau': [0.001, 0.01, 0.1],
                        'eta_1': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                        'eta_2': [0.1, 0.5, 1.],
                        'eta_3': [5, 10, 20]}
    elif method == 'ar':
        search_space = {'rho0': [2.5, 5.0, 7.5, 10.0],
                        'up':[5.0, 10.0],
                        'low':[-10.0, -5.0],
                        'ent_weight': [0.01, 0.1, 1.0]}
    elif method == 'ba3us':
        search_space = {'cot_weight': [0.1, 0.5, 1, 5, 10], 'ent_weight': [0.01, 0.05, 0.1, 0.5, 1]}
    elif method == 'mpot':
        search_space = {'epsilon': [0.5, 1.0, 1.5],
                        'eta_1': [0.0001, 0.001, 0.01, 0.1, 1.0],
                        'eta_2': [0.1, 1.0, 5.0, 10.0],
                        'mass': [0.1, 0.2, 0.3, 0.4]}
    elif method == 'safn':
        search_space = {'lambda': [0.005, 0.01, 0.05, 0.1, 0.5], 'delta_r': [0.01, 0.1, 1.0]}
    elif method == 'pada':
        search_space = {'lambda': [0.1, 0.5, 1.0, 5.0, 10.0]}
    elif method == 'etn':
        search_space = {'adv_loss_tradeoff': [0.1, 0.5, 1.0, 5.0], # lambda_1
                        'entropy_tradeoff': [0.05, 0.1, 0.2, 0.5], # lambda_2
                        'adv_loss_aug_tradeoff': [0.5, 1.0, 5.0, 10.0], # lambda_3
                        'ce_aug_tradeoff': [0.1, 0.5, 1.0, 5.0], # lambda_4
                       }
    elif method == 'source_only_plus':
        search_space = {}
    search_space['radius'] = [5.0, 10.0, 20.0]
    return search_space