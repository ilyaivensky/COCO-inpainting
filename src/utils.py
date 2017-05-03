#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os

import numpy as np
from theano import config

import logging.config
import yaml

def setup_logging(
        
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'):
    
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        
def generate_z(shape):
    
    return np.asarray(np.random.uniform(size=shape), dtype=config.floatX)
          
def show_samples(ids, target, samples, captions, vocab_idx, model_name, batch_id, out_dir):

    import math
    import matplotlib.pyplot as plt
    
    file_name = '-'.join([model_name, '{:03d}'.format(batch_id)])
    
    nb_samples = len(samples)
    
    ncol = 10
    nrows = math.ceil(nb_samples / ncol) * 2
    
    num_real = 0
    
    for i in range(nb_samples):
            
        if captions[i].nnz == 0:
            logging.warning('Empty captions for {}'.format(ids[i]))
             
        ax = plt.subplot(nrows, ncol, num_real + 1)
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        plt.imshow(target[i])
    
        ax = plt.subplot(nrows, ncol, num_real + ncol + 1)
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        plt.imshow(samples[i])
        
        num_real += 1
        if num_real % ncol == 0:
            num_real += ncol
    
    plt.suptitle(file_name)   
    plt.savefig(os.path.join(out_dir, '.'.join([file_name,'png'])))
 #   plt.show()