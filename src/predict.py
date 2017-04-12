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
        
        model.frames_var.set_value((lu.floatX(frames) / 255).transpose(0,3,1,2))
        model.noise_var.set_value(lu.floatX(np.random.randn(len(imgs),100)))
        model.caps_var.set_value(sparse_floatX(caps))
    
     
        samples, loss = model.generate(0,batch_size)
        show_samples(idxs, imgs, (samples.transpose(0,2,3,1) * 255).astype(np.uint8), caps, vocab_idx)
        predict_loss += loss
        
        break

    logging.info("  prediction loss:\t\t{}".format(predict_loss))

def main(data_file, params_file, w2v_file):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))  
    
    batch_size=30
    
    gan = GAN(batch_size, 1, 11172)     
    gan.load_params(params_file)
    
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