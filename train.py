import os, random, argparse
import numpy as np
import torch
from utils.logger import *
from utils.model_selection import *

def train(algorithm):
    
    log_file = Logger(algorithm.logger_hp)
    if algorithm.logger_hp is not None:
        log_file.write(f'Logger Hyper-Parameters', time=False)
        for key in algorithm.logger_hp:
            log_file.write(f'    {key}: {algorithm.logger_hp[key]}', time=False)
    if algorithm.net_hp is not None:
        log_file.write(f'Net Hyper-Parameters', time=False)
        for key in algorithm.net_hp:
            log_file.write(f'    {key}: {algorithm.net_hp[key]}', time=False)
    if algorithm.train_hp is not None:
        log_file.write(f'Training Hyper-Parameters', time=False)
        for key in algorithm.train_hp:
            log_file.write(f'    {key}: {algorithm.train_hp[key]}', time=False)
    if algorithm.dset_hp is not None:
        log_file.write(f'Dataset Hyper-Parameters', time=False)
        for key in algorithm.dset_hp:
            log_file.write(f'    {key}: {algorithm.dset_hp[key]}', time=False)
    if algorithm.loss_hp is not None:
        log_file.write(f'Loss Hyper-Parameters', time=False)
        for key in algorithm.loss_hp:
            log_file.write(f'    {key}: {algorithm.loss_hp[key]}', time=False)
            
    algorithm.set_dsets()
    algorithm.set_dsets_model_selection()
    algorithm.set_dset_loaders()
    algorithm.set_dset_loaders_model_selection()
    algorithm.set_base_network()
    algorithm.prep_for_train()
    
    log_results = init_log_results(
        algorithm.train_hp['test_interval'], 
        algorithm.train_hp['max_iterations'], 
        algorithm.net_hp['class_num'],
        algorithm.logger_hp['save_outputs_evl'],
        algorithm.logger_hp['model_selection_metrics'])
    
    best_values = {}
    for metric in algorithm.logger_hp['model_selection_metrics']:
        if ('acc' in key) or ('snd' == key):
            best_values[metric] = 0.0
        else:
            best_values[metric] = np.Inf

    if algorithm.__class__.__name__ == 'SourceOnlyPlus':
        gamma_acc = []

    log_file.write("Started Training")
    for i in range(algorithm.train_hp['max_iterations'] + 1):
        if (i % algorithm.train_hp['test_interval'] == 0 and i > 0) or (i == algorithm.train_hp['max_iterations']):
            if algorithm.dset_hp['use_val']:
                s_features, s_logits, s_labels = get_data(algorithm.dset_loaders['source_train'], algorithm.base_network)
                v_features, v_logits, v_labels = get_data(algorithm.dset_loaders['source_val'], algorithm.base_network)
            t_features, t_logits, t_labels = get_data(algorithm.dset_loaders['test'], algorithm.base_network)

            algorithm.class_weight = get_class_weight(t_logits)

            current_values = {}
            
            for metric in algorithm.logger_hp['model_selection_metrics']:
                if metric == 't_acc':
                    current_values[metric] = get_acc(t_logits, t_labels)
                elif metric == 'ent':
                    current_values[metric] = get_mean_ent(t_logits)
                elif metric == 'snd':
                    current_values[metric] = get_snd(t_logits)
                elif metric == 's_acc':
                    current_values[metric] = get_acc(v_logits, v_labels)
                elif metric == 'dev_lr':
                    weights = get_importance_weights_lr(s_features, t_features, v_features, algorithm.train_hp['seed'])
                    error = get_error(v_logits, v_labels)
                    current_values[metric] = get_dev_risk(weights, error)
                elif metric == 'dev_mlp':
                    weights = get_importance_weights_mlp(s_features, t_features, v_features, algorithm.train_hp['seed'])
                    error = get_error(v_logits, v_labels)
                    current_values[metric] = get_dev_risk(weights, error)
                elif metric == 'dev_svm':
                    weights = get_importance_weights_svm(s_features, t_features, v_features, algorithm.train_hp['seed'])
                    error = get_error(v_logits, v_labels)
                    current_values[metric] = get_dev_risk(weights, error)
                elif metric == '1shot_acc':
                    _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_1shot'], algorithm.base_network)
                    current_values[metric] = get_acc(temp_logits, temp_labels)
                elif metric == '1shot_10crop_acc':
                    current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_1shot_10crop'], algorithm.base_network)
                elif metric == '3shot_acc':
                    _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_3shot'], algorithm.base_network)
                    current_values[metric] = get_acc(temp_logits, temp_labels)
                elif metric == '3shot_10crop_acc':
                    current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_3shot_10crop'], algorithm.base_network)
                elif metric == '25random_acc':
                    _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_25random'], algorithm.base_network)
                    current_values[metric] = get_acc(temp_logits, temp_labels)
                elif metric == '25random_10crop_acc':
                    current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_25random_10crop'], algorithm.base_network)
                elif metric == '50random_acc':
                    _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_50random'], algorithm.base_network)
                    current_values[metric] = get_acc(temp_logits, temp_labels)
                elif metric == '50random_10crop_acc':
                    current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_50random_10crop'], algorithm.base_network)
                elif metric == '100random_acc':
                    _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_100random'], algorithm.base_network)
                    current_values[metric] = get_acc(temp_logits, temp_labels)
                elif metric == '100random_10crop_acc':
                    current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_100random_10crop'], algorithm.base_network)
                else:
                    raise NotImplementedError

            for metric in current_values:
                if ('acc' in key) or ('snd' == key):
                    if current_values[metric] > best_values[metric]:
                        best_values[metric] = current_values[metric]
                        if algorithm.logger_hp['save_models']:
                            algorithm.save_model(f'model_{metric}.pt')
                else:
                    if current_values[metric] < best_values[metric]:
                        best_values[metric] = current_values[metric]
                        if metric in algorithm.logger_hp['save_models']:
                            algorithm.save_model(f'model_{metric}.pt')
                        
            log_temp = {'iterations': i, 'class_weights': algorithm.class_weight.cpu()}
        
            for metric in current_values:
                log_temp[metric] = current_values[metric]
        
            # update log_results
            log_results = update_log_results(log_results, log_temp)
            
            if algorithm.logger_hp['save_outputs_evl']:
                log_results = update_log_results_outputs(log_results, i, t_logits, t_labels)

            update_log_file(log_results, log_file)
            
            
            if algorithm.__class__.__name__ == 'SourceOnlyPlus':
                thrs = np.linspace(0,1,101)[:-1]
                temp_accs = []
                for thr in thrs:
                    softmaxes = torch.nn.Softmax(dim=1)(t_logits)
                    pmax = softmaxes.max(dim=1).values
                    temp_weights = softmaxes[pmax>=thr].mean(dim=0)
                    temp_accs.append((torch.sum((softmaxes*temp_weights).argmax(dim=1) == t_labels)/softmaxes.shape[0]))
                    
                gamma_acc.append(temp_accs)
            
            if i == algorithm.train_hp['max_iterations']:
                break
        
        
        algorithm.update_dsets(i)
        algorithm.optimizer = algorithm.lr_scheduler(algorithm.optimizer, i, **algorithm.schedule_param)        
        algorithm.update_dset_loaders(i)
        total_loss, clf_loss, adpt_loss = algorithm.update(i)

        # Logging loss
        log_results['total_loss'].append(total_loss.item())
        log_results['clf_loss'].append(clf_loss.item())
        log_results['adpt_loss'].append(adpt_loss.item())

    if algorithm.__class__.__name__ == 'SourceOnlyPlus':
        log_results['t_acc_gamma'] = gamma_acc
    # Save final model
    if algorithm.logger_hp['save_model_final']:
        algorithm.save_model('model_final.pt')
    np.save(os.path.join(algorithm.logger_hp['output_dir'], 'results.npy'), log_results)
    log_file.write('Finished Training')