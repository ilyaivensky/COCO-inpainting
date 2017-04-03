#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np

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

def iterate_minibatches(hfp, split, batchsize, shuffle=False):
        
    x = hfp['/%s/x' % split]
    z = hfp['/%s/z' % split]
    y = hfp['/%s/y' % split]
    
    assert len(x) == len(y)
    assert len(x) == len(z)

    if shuffle:
        indices = np.arange(len(x))
        np.random.shuffle(indices)
    for start_idx in range(0, len(x) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield x[excerpt], z[excerpt], y[excerpt]
            
   
        
def show_samples(target, samples):

    nb_samples = len(samples)
    s = (samples.transpose(0,2,3,1) * 255).astype(np.uint8)
    
    for i in range(nb_samples):
            
        plt.subplot(2, nb_samples, i+1)
        plt.imshow((target[i] * 255).astype(np.uint8))
    
        img_pred = np.copy(s[i])
        plt.subplot(2, nb_samples, nb_samples+i+1)
        plt.imshow(img_pred)
    
    plt.show()