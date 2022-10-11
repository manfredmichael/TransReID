# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .defaults import _C as cfg
from .defaults import _C as cfg_test

def get_model_hyperparameters(cfg):
    params = {}
    for key in cfg.MODEL.keys():
        params[key] = cfg.MODEL[key]

    return params
