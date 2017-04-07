#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

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

class DataIterator(object):
    
    def __init__(self, hfp, split):
        
        self.logger = logging.getLogger('DataIterator')
        
        self.ids = hfp['/%s/id' % split]
        
        self.x = hfp['/%s/frame' % split]
        self.y = hfp['/%s/img' % split]
    
        capt_grp = hfp['/%s/capt' % split]
        capt_matrix_shape = (capt_grp.attrs['shape0'], capt_grp.attrs['shape1'])
        
        capt_data = hfp['/%s/capt/data' % split]
        capt_indices = hfp['/%s/capt/indices' % split]
        capt_indptr = hfp['/%s/capt/indptr' % split]
    
        self.captions = sparse.csr_matrix((capt_data, capt_indices, capt_indptr), shape=capt_matrix_shape)
        self.logger.info('Reconstructed one-hot matrix of captions, shape=({},{})'.format(self.captions.shape[0], self.captions.shape[1]))
    
        assert len(self.x) == len(self.y)
        
    def iterate_minibatches(self, batchsize, shuffle=False):
            
        if shuffle:
            indices = np.arange(len(self.x))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.x) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
                        
            yield self.ids[excerpt], self.x[excerpt], self.y[excerpt], self.captions[excerpt].toarray()
            
          
def show_samples(id, target, samples, captions, vocab_idx):

    import math
    
    nb_samples = len(samples)
    
    ncol = 10
    nrows = math.ceil(nb_samples / ncol) * 2
    
    num_real = 0

    for i in range(nb_samples):
            
        indices = [idx for idx, e in enumerate(captions[i]) if e != 0]
        if not indices:
            logging.warning('Empty captions for {}'.format(id[i]))
             
        plt.subplot(nrows, ncol, num_real + 1)
        plt.imshow(target[i])
    
        plt.subplot(nrows, ncol, num_real + ncol + 1)
        plt.imshow(samples[i])
        
        num_real += 1
        if num_real % ncol == 0:
            num_real += ncol
            
    plt.show()