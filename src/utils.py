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
          
def show_samples(id, target, samples, captions, loss, vocab_idx, model_name, out_dir):

    import math
    import matplotlib.pyplot as plt
    
    nb_samples = len(samples)
    
    ncol = 10
    nrows = math.ceil(nb_samples / ncol) * 2
    
    num_real = 0
    
    for i in range(nb_samples):
            
        if captions[i].nnz == 0:
            logging.warning('Empty captions for {}'.format(id[i]))
             
        ax = plt.subplot(nrows, ncol, num_real + 1)
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        plt.imshow(target[i])
    
        ax = plt.subplot(nrows, ncol, num_real + ncol + 1)
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False) 
        ax.set_title('{:.3f}'.format(loss[i]))
        plt.imshow(samples[i])
        
        num_real += 1
        if num_real % ncol == 0:
            num_real += ncol
    
    plt.suptitle(model_name)   
    plt.savefig(os.path.join(out_dir, '.'.join([model_name, 'png'])))
 #   plt.show()