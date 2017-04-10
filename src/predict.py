#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from lasagne import utils as lu

import gensim

import numpy as np
from numpy import random as rnd 

from GAN import GAN
from utils import setup_logging, show_samples

import argparse
import logging

from dataset import H5PYSparseDataset
from fuel.schemes import SequentialScheme

from sparse_matrix_utils import sparse_floatX

def predict(model, data_file, split, w2v_model, batch_size):
      
    vocab_idx = w2v_model.wv.index2word
    
    logging.info("Starting predicting...")
    
    data = H5PYSparseDataset(
        data_file, 
        (split,), 
        sources=('val2014/id', 'val2014/frame', 'val2014/img', 'val2014/capt'), 
        load_in_memory=True)
    
    data.example_iteration_scheme = SequentialScheme(data.num_examples, batch_size)
    
    predict_loss = 0
    
    data_stream = data.get_example_stream()
    
    for idxs, frames, imgs, caps in data_stream.get_epoch_iterator():
        
        x_var = (lu.floatX(frames) / 255).transpose(0,3,1,2)
        caps_var = sparse_floatX(caps)
        
        noise_var = lu.floatX(np.random.randn(len(x_var),100))
        
        samples, loss = model.predict(x_var, caps_var, noise_var)
        show_samples(idxs, imgs, (samples.transpose(0,2,3,1) * 255).astype(np.uint8), caps, vocab_idx)
        predict_loss += loss
        
        break

    logging.info("  prediction loss:\t\t{}".format(predict_loss))

def main(data_file, params_file, w2v_file):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))  
    
    gan = GAN()     
    gan.load_params(params_file)
    
    batch_size=30
    
    w2v_model = gensim.models.Word2Vec.load(w2v_file)
    
#    with h5py.File(data_file,'r') as hf:
    predict(gan, data_file, 'val2014', w2v_model, batch_size)

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