#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from lasagne import utils as lu

import gensim

import numpy as np
from numpy import random as rnd 

from DCGAN import DCGAN
from utils import DataIterator, setup_logging
from utils import show_samples

import h5py
import argparse
import logging

def predict(model, data_fp, split, w2v_model, batch_size):
      
    vocab_idx = w2v_model.wv.index2word
    
    logging.info("Starting predicting...")
    
    iterator = DataIterator(data_fp, split)
   
    predict_loss = 0
    nr_batches = 0
    
    for batch in iterator.iterate_minibatches(batch_size, shuffle=False):
        ids, x, y, captions = batch
        
        print captions
        
        x_var = lu.floatX(x) / 255
        y_var = lu.floatX(y) / 255
        
        samples, loss = model.predict(x_var, y_var, batch_size)
        show_samples(ids, y, (samples.transpose(0,2,3,1) * 255).astype(np.uint8), captions, vocab_idx)
        predict_loss += loss
        nr_batches += 1
        
        break

    logging.info("  prediction loss:\t\t{}".format(predict_loss / nr_batches))

def main(data_file, params_file, w2v_file):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))  
    
    gan = DCGAN()     
    gan.load_params(params_file)
    
    batch_size=30
    
    w2v_model = gensim.models.Word2Vec.load(w2v_file)
    
    with h5py.File(data_file,'r') as hf:
        predict(gan, hf, 'val2014', w2v_model, batch_size)

if __name__ == '__main__':
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    rnd.seed(87537)
    
    parser = argparse.ArgumentParser(description='Tests predictions of trained DCGAN on COCO val2014 dataset')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-m','--model', type=str, help='model name')
    parser.add_argument('-w', '--w2v_file', type=str, default='../models/word2vec.512.model', help='word2vec model file')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
    
    main(args.data_file, args.model, args.w2v_file)