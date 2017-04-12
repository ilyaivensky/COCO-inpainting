#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
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
          
def show_samples(id, target, samples, captions, vocab_idx):

    import math
    import matplotlib.pyplot as plt
    
    nb_samples = len(samples)
    
    ncol = 10
    nrows = math.ceil(nb_samples / ncol) * 2
    
    num_real = 0
    
    for i in range(nb_samples):
            
        if captions[i].nnz == 0:
            logging.warning('Empty captions for {}'.format(id[i]))
             
        plt.subplot(nrows, ncol, num_real + 1)
        plt.imshow(target[i])
    
        plt.subplot(nrows, ncol, num_real + ncol + 1)
        plt.imshow(samples[i])
        
        num_real += 1
        if num_real % ncol == 0:
            num_real += ncol
            
    plt.show()