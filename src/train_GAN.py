#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import sys

import theano

import lasagne

import numpy as np

from GAN import GAN
from utils import setup_logging

import h5py

import argparse
import time

import logging


from dataset import H5PYSparseDataset
from fuel.schemes import SequentialScheme
#from fuel.transformers import Transformer
#from PIL import Image

from sparse_matrix_utils import sparse_floatX

#from predict import predict

def train(data_file, out_model, voc_size, num_epochs, batch_size, batches_on_gpu, batches_to_train=None, 
         split='train2014', initial_eta=2e-4, unroll=1, params_file=None):
    
    logger = logging.getLogger(__name__)
    
        
    theano.config.floatX = 'float32'
    theano.config.exception_verbosity='high'
    theano.config.optimizer='fast_compile'
    
    np.random.seed(87537)
    
    logger.info('Loading data from {}...'.format(data_file))
    
    data = H5PYSparseDataset(
        data_file, 
        (split,), 
        sources=('train2014/frame', 'train2014/img', 'train2014/capt'), 
        load_in_memory=True)
    
    data.example_iteration_scheme = SequentialScheme(data.num_examples, batch_size * batches_on_gpu)
    
    gan = GAN(batch_size, batches_on_gpu, voc_size)
    if params_file is not None:
        gan.load_params(params_file)
    
    logging.info('Starting training: num_epochs={}, batches_to_train={}, batches_on_gpu={}, unroll={}, data_file={}'.format(num_epochs, batches_to_train, batches_on_gpu, unroll, data_file))
   
    data_stream = data.get_example_stream()
    
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_D_loss = 0
        train_G_loss = 0
        train_batches = 0
        
        data_stream.reset()
          
        start_time = time.time()
          
        for gpu_e, (frames, imgs, caps) in enumerate(data_stream.get_epoch_iterator()):
        
            logging.debug('{}: Loading {} examples to GPU'.format(gpu_e+1, len(imgs)))
            
            gan.frames_var.set_value((lasagne.utils.floatX(frames) / 255).transpose(0,3,1,2))
            gan.img_var.set_value((lasagne.utils.floatX(imgs) / 255).transpose(0,3,1,2))
            gan.noise_var.set_value(lasagne.utils.floatX(np.random.randn(len(imgs),100)))
            gan.caps_var.set_value(sparse_floatX(caps))
            
            logging.debug('Done loading {} examples to GPU'.format(len(imgs)))
            logging.debug('frames.shape={}, imgs.shape={}, caps.shape={}'.format(frames.shape, imgs.shape, caps.shape))
            
            fi = 0
                
            for i in range(batches_on_gpu):
       
                train_batches += 1
                """
                Train discriminator on real images right away, but delay training on fake ones
                Accumulate all minibatches, and then train discriminator and generator on fake images
                """ 
              #  logging.debug('real, first={}, last={}'.format(i*batch_size, (i+1)*batch_size))
                train_D_loss += gan.train_D_real(i*batch_size, (i+1)*batch_size)
                  
                acc_idx = train_batches % unroll
               
                if acc_idx == 0 or i == batches_on_gpu-1:
                    # train generator with accumulated batches
                    while fi <= i: 
                     #   logging.debug('fake, first={}, last={}'.format(fi*batch_size, (fi+1)*batch_size))
                        train_D_loss += gan.train_D_fake(fi*batch_size, (fi+1)*batch_size)
                        train_G_loss += gan.train_G(fi*batch_size, (fi+1)*batch_size)
                        fi+=1
                    
            if (batches_to_train is not None) and (train_batches >= batches_to_train):
                break
        
  
        # Then we print the results for this epoch:
        logging.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logging.info("  training loss (D/G):\t\t{}".format(np.array([train_D_loss, train_G_loss]) / train_batches))
 
        # Be on a safe side - if the job is killed, it is better to preserve at least something
        if epoch % 10 == 9 or epoch == num_epochs - 1:
            gan.save_params('{}.{}'.format(out_model, epoch + 1))
              
#    predict(gan, data_fp, 'val2014', 10)
        

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Trains a DCGAN on COCO using Lasagne')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-n', '--num_epochs', type=int, default=1000, help='number of epochs (default: 1000)')
    parser.add_argument('-u', '--unroll', type=int, default=None, help='unroll (num mini-batches) (default=None)')
    parser.add_argument('-p', '--params_dir', type=str, help='directory with parameters files (npz format)')
    parser.add_argument('-b', '--batches_to_train', type=int, help='the total max number of batches to train (defailt: None, meaning train all batches). If provided, it will be multiplied by delay_g_training')
    parser.add_argument('-s', '--batch_size', type=int, default=128, help='the number of examples per batch')
    parser.add_argument('-o', '--out_model', type=str, default='../models/GAN', help='otput model')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    parser.add_argument('-g', '--batches_on_gpu', type=int, default=10, help='number of mini-batches to load simultaneously on GPU')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
 
    train(args.data_file, out_model=args.out_model, voc_size=11172,  num_epochs=args.num_epochs, batch_size=args.batch_size,
         params_file=args.params_dir, batches_on_gpu=args.batches_on_gpu, unroll=args.unroll, 
         batches_to_train=args.batches_to_train)
