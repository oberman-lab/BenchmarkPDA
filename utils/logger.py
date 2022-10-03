import os
from datetime import datetime
import numpy as np
import torch

class Logger():
    def __init__(self, logger_hp: dict):
        self.file_path = os.path.join(logger_hp['output_dir'], f"{logger_hp['filename']}.txt")
        self.printQ = logger_hp['printQ']
        f = open(self.file_path, 'w')
        f.close()
        
    def write(self, string: str, time: bool = True):
        f = open(self.file_path, 'a')
        f.write(f'{string}\n')
        f.close()
        if self.printQ:
            if time:
                time_preamble = datetime.now().strftime('%d/%m/%Y %H:%M:%S ')
            else:
                time_preamble = ''
            print(f'{time_preamble}{string}')

def update_log_file(log_results: dict, log_file: Logger):
    i = log_results['iterations'][-1]
    current_total_loss = np.mean(log_results['total_loss'][-log_results['test_interval']:])
    current_clf_loss = np.mean(log_results['clf_loss'][-log_results['test_interval']:])
    current_adpt_loss = np.mean(log_results['adpt_loss'][-log_results['test_interval']:])
    to_print = f'Iter [{i:{len(str(log_results["max_iterations"]))}d}/{log_results["max_iterations"]}]'
    to_print += f' Loss: {current_total_loss:1.2f}'
    if 't_acc' in log_results:
        to_print += f' Accuracy Target: {log_results["t_acc"][-1]*100:1.2f}'
    log_file.write(to_print)


def init_log_results(test_interval: int, max_iterations: int, class_num: int, save_outputs: bool, model_selection_metrics: dict):
    log_results = {}
    log_results['test_interval'] = test_interval
    log_results['max_iterations'] = max_iterations
    log_results['iterations'] = []
    log_results['class_weights'] = np.zeros((max_iterations//test_interval, class_num))
    log_results['clf_loss'] = []
    log_results['adpt_loss'] = []
    log_results['total_loss'] = []
    for metric in model_selection_metrics:
        log_results[metric] = []
        
    if save_outputs:
        log_results['logits'] = None
        log_results['labels'] = None
    return log_results


def update_log_results(log_results: dict, log_temp: dict):
    for key in log_temp.keys():
        if 'class_weights' == key:
            ix = (log_results['iterations'][-1]//log_results['test_interval'])-1
            log_results['class_weights'][ix, :] = log_temp['class_weights']
        else:
            log_results[key].append(log_temp[key])
    return log_results

def update_log_results_outputs(log_results: dict, i: int, logits, labels):
    if log_results['labels'] is None:
        log_results['labels'] = labels.detach().cpu()
    if log_results['logits'] is None:
        log_results['logits'] = torch.zeros((log_results['max_iterations']//log_results['test_interval'], logits.shape[0], logits.shape[1]))
    log_results['logits'][(i//log_results['test_interval'])-1, :, :] = logits
    return log_results

