#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import theano

import gensim

import numpy as np
from numpy import random as rnd 

from DCGAN import DCGAN
from utils import setup_logging, show_samples, generate_z

import argparse
import logging

from dataset import H5PYSparseDataset
from fuel.schemes import SequentialScheme
from fuel.transformers.defaults import uint8_pixels_to_floatX
import fuel

from sparse_matrix_utils import sparse_floatX

import  os
from os.path import basename

def evaluate(model, data_file, split, w2v_model, num_batches, model_name, out_dir, nrows, ncols):
      
    vocab_idx = w2v_model.wv.index2word
    
    model_dir = os.path.join(out_dir, model_name)
    
    try:
        os.mkdir(model_dir)
    except OSError:
        pass
        
    
    logging.info("Starting predicting...")
    
    data = H5PYSparseDataset(
        data_file, 
        (split,), 
        sources=('{}/id'.format(split), '{}/frame'.format(split), '{}/img'.format(split), '{}/capt'.format(split)), 
        load_in_memory=True)
    
    batch_size = nrows * ncols
    num_examples = num_batches * batch_size
    data.example_iteration_scheme = SequentialScheme(num_examples, batch_size) #we intend to do only 1 batch
    data.default_transformers = uint8_pixels_to_floatX(('{}/frame'.format(split), '{}/img'.format(split)))
    
    data_stream = data.apply_default_transformers(
        data.get_example_stream())
    
    for batch_id, (idxs, frames, imgs, caps) in enumerate(data_stream.get_epoch_iterator()):
        
        model.img_var.set_value(imgs.transpose(0,3,1,2))
        model.frames_var.set_value(frames.transpose(0,3,1,2))
        model.noise_var.set_value(generate_z((len(imgs),model.generator.noise_shape)))
        model.caps_var.set_value(sparse_floatX(caps))
    
        samples, loss, acc = model.evaluate_fake(0,batch_size)
        show_samples(idxs, (imgs * 255).astype(np.uint8), (samples.transpose(0,2,3,1) * 255).astype(np.uint8), caps, vocab_idx, model_name, batch_id, model_dir, nrows, ncols, split)
         
        logging.info('prediction loss and acc:\t\t{}'.format(zip(loss, acc)))
        logging.info('loss mean: {:.3f}, var: {:.3f}'.format(np.asscalar(np.mean(loss, axis=0)), np.asscalar(np.var(loss, axis=0))))
        logging.info('avg acc: {:.3f}'.format(np.sum(acc) / batch_size))
        
        if batch_id == num_batches - 1:
            break

def main(data_file, params_file, w2v_file, out_dir, num_batches, nrows, ncols, split):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))  
    
    gan = DCGAN(nrows * ncols, 11172)     
    gan.load_params(params_file)
    
    w2v_model = gensim.models.Word2Vec.load(w2v_file)
    evaluate(gan, data_file, split, w2v_model, num_batches, basename(params_file), out_dir, nrows, ncols)

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
    parser.add_argument('-n', '--num_batches', type=int, default=1, help='number of mini-batches')
    parser.add_argument('-r', '--nrows', type=int, default=3, help='number of double rows (origin and sample)')
    parser.add_argument('-c', '--ncols', type=int, default=6, help='number of columns')
    parser.add_argument('--split', type=str, default='val', help='split of dataset (train or val)')
    
    args = parser.parse_args()
    
    if args.split == 'val':
        split = 'val2014'
    elif args.split == 'train':
        split = 'train2014'
    else:
        raise ValueError('Invalid split')   
    
    setup_logging(default_path=args.log_file)
    
    main(args.data_file, args.model, args.w2v_file, args.out_dir, args.num_batches, args.nrows, args.ncols, split)
