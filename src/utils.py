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
          
def show_samples(ids, target, samples, captions, vocab_idx, model_name, batch_id, out_dir, nrows, ncols, split):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    file_name = '-'.join([model_name, split, '{:03d}'.format(batch_id)])

    subplot_num = 0
    
    plt.figure(figsize = (nrows * 2, ncols))
    gs1 = gridspec.GridSpec(nrows * 2, ncols)
    gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

   
    for i in range(nrows * ncols):
         
        """
        show target
        """   
        ax = plt.subplot(gs1[subplot_num])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(target[i])

        """
        show sample
        """ 
        ax = plt.subplot(gs1[subplot_num + ncols])
        plt.axis('off')
        ax.set_aspect('equal')
        plt.imshow(samples[i])
         
        subplot_num += 1
        if subplot_num % ncols == 0:
            subplot_num += ncols
    
    plt.suptitle(file_name)   
    plt.savefig(os.path.join(out_dir, '.'.join([file_name,'png'])))
 #   plt.show()