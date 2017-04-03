#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from lasagne import utils as lu

import numpy as np
from numpy import random as rnd 

from DCGAN import DCGAN
from utils import iterate_minibatches, setup_logging
from utils import show_samples

import h5py
import argparse
import logging

def predict(model, data_fp, split, batch_size):
      
    logging.info("Starting predicting...")
   
    predict_loss = 0
    nr_batches = 0
    
    for batch in iterate_minibatches(data_fp, split, batch_size, shuffle=False):
        x, _, y = batch
        
        x_var = lu.floatX(x) / 255
        y_var = lu.floatX(y) / 255
        
        samples, loss = model.predict(x_var, y_var, batch_size)
        show_samples(y, (samples.transpose(0,2,3,1) * 255).astype(np.uint8))
        predict_loss += loss
        nr_batches += 1
        
        break

    logging.info("  prediction loss:\t\t{}".format(predict_loss / nr_batches))

def main(data_file, params_file):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))  
    
    gan = DCGAN()     
    gan.load_params(params_file)
    
    batch_size=30
    
    with h5py.File(data_file,'r') as hf:
        predict(gan, hf, 'val2014', batch_size)

if __name__ == '__main__':
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    rnd.seed(87537)
    
    parser = argparse.ArgumentParser(description='Tests predictions of trained DCGAN on COCO val2014 dataset')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-m','--model', type=str, help='model name')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
    
    main(args.data_file, args.model)