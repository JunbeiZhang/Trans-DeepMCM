# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import configparser
import torch

from lifelines.utils import concordance_index

import configparser
import ast

def read_config(ini_file):
    ''' Performs read config file and parses it.

    :param ini_file: (String) the path of a .ini file.
    :return config: (dict) the dictionary of information in ini_file.
    '''
    def _build_dict(items):
        result = {}
        for item in items:
            try:
                result[item[0]] = ast.literal_eval(item[1])
            except (ValueError, SyntaxError):
                result[item[0]] = item[1]  # Keep the value as a string if it can't be evaluated
        return result

    # create configparser object
    cf = configparser.ConfigParser()
    # read .ini file
    cf.read(ini_file, encoding='utf-8')
    config = {sec: _build_dict(cf.items(sec)) for sec in cf.sections()}
    return config


def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def create_logger(logs_dir):
    ''' Performs creating logger

    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger

import torch

def integral_brier_score(model, dataset, t_max, device='cuda'):
    ''' Calculates the Integral Brier Score (IBS) for survival prediction models using PyTorch.

    :param model: (DeepSurv) trained survival model
    :param dataset: (SurvivalDataset) dataset with survival data
    :param t_max: (float) maximum time to calculate IBS up to
    :param device: (str) computation device ('cpu' or 'cuda')
    :return ibs: (float) the integral brier score calculated over time up to t_max
    '''
    model.eval()  # Set the model to evaluation mode
    ibs_sum = 0.0
    n = len(dataset)
    times = torch.linspace(0, t_max, steps=100).to(device)  # time points for evaluation

    with torch.no_grad():  # Disable gradient calculation
        for i in range(n):
            x_tensor, y_tensor, e_tensor = dataset[i]
            x_tensor, y_tensor, e_tensor = x_tensor.to(device), y_tensor.to(device), e_tensor.to(device)

            # Get the second output (h_pop) as risk prediction
            _, risk_pred = model(x_tensor.unsqueeze(0))

            # Calculate Brier score for each time point
            for t in times:
                mask_t = y_tensor > t
                surv_pred = torch.sigmoid(-risk_pred + t)  # Convert risk to survival probability
                brier_score = torch.mean((mask_t.float() - surv_pred) ** 2)
                ibs_sum += brier_score.item() * (t / t_max)

    # Normalize by the number of observations and the maximum time
    ibs = ibs_sum / (n * len(times))
    return ibs
