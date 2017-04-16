#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import theano

import gensim

import numpy as np
from numpy import random as rnd 

from GAN import GAN
from utils import setup_logging, show_samples, generate_z

import argparse
import logging

from dataset import H5PYSparseDataset
from fuel.schemes import SequentialScheme
from fuel.transformers.defaults import uint8_pixels_to_floatX
import fuel

from sparse_matrix_utils import sparse_floatX

from os.path import basename

def evaluate(model, data_file, split, w2v_model, batch_size, model_name, out_dir):
      
    vocab_idx = w2v_model.wv.index2word
    
    logging.info("Starting predicting...")
    
    data = H5PYSparseDataset(
        data_file, 
        (split,), 
        sources=('val2014/id', 'val2014/frame', 'val2014/img', 'val2014/capt'), 
        load_in_memory=True)
    
    data.example_iteration_scheme = SequentialScheme(batch_size, batch_size) #we intend to do only 1 batch
    data.default_transformers = uint8_pixels_to_floatX(('val2014/frame', 'val2014/img'))
    
    data_stream = data.apply_default_transformers(
        data.get_example_stream())
    
    for idxs, frames, imgs, caps in data_stream.get_epoch_iterator():
        
        model.img_var.set_value(imgs.transpose(0,3,1,2))
        model.frames_var.set_value(frames.transpose(0,3,1,2))
        model.noise_var.set_value(generate_z((len(imgs),model.generator.noise_shape)))
        model.caps_var.set_value(sparse_floatX(caps))
    
        samples, loss, acc = model.evaluate_fake(0,batch_size)
        show_samples(idxs, (imgs * 255).astype(np.uint8), (samples.transpose(0,2,3,1) * 255).astype(np.uint8), caps, loss.ravel(), vocab_idx, model_name, out_dir)
         
        logging.info('prediction loss and acc:\t\t{}'.format(zip(loss, acc)))
        logging.info('loss mean: {:.3f}, var: {:.3f}'.format(np.asscalar(np.mean(loss, axis=0)), np.asscalar(np.var(loss, axis=0))))
        logging.info('avg acc: {:.3f}'.format(np.sum(acc) / batch_size))
        
        break

def main(data_file, params_file, w2v_file, out_dir):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))  
    
    batch_size=30
    
    gan = GAN(batch_size, 11172)     
    gan.load_params(params_file)
    
    w2v_model = gensim.models.Word2Vec.load(w2v_file)
    
#    with h5py.File(data_file,'r') as hf:
    evaluate(gan, data_file, 'val2014', w2v_model, batch_size, basename(params_file), out_dir)

if __name__ == '__main__':
    
    theano.config.floatX = 'float32'
    fuel.config.floatX = theano.config.floatX
    
    theano.exception_verbosity='high'
    
    rnd.seed(87537)
    
    parser = argparse.ArgumentParser(description='Tests predictions of trained DCGAN on COCO val2014 dataset')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-m','--model', type=str, help='model name')
    parser.add_argument('-w', '--w2v_file', type=str, default='../models/word2vec.512.model', help='word2vec model file')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    parser.add_argument('-o', '--out_dir', type=str, default='../results', help='output directory')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
    
    main(args.data_file, args.model, args.w2v_file, args.out_dir)