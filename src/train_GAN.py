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
import math


from dataset import H5PYSparseDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Transformer
from fuel.transformers.defaults import uint8_pixels_to_floatX
import fuel
#from PIL import Image

from sparse_matrix_utils import sparse_floatX

#from predict import predict

def train(data_file, out_model, out_freq, voc_size, num_epochs, batch_size, batches_on_gpu, max_example_stream_iter=None, 
         split='train2014', initial_eta=2e-4, unroll=1, params_file=None):
    
    logger = logging.getLogger(__name__)
    
        
    theano.config.floatX = 'float32'
    fuel.config.floatX = theano.config.floatX
#     theano.config.exception_verbosity='high'
#     theano.config.optimizer='fast_compile'
    
    np.random.seed(87537)
    
    logger.info('Loading data from {}...'.format(data_file))
    
    data = H5PYSparseDataset(
        data_file, 
        (split,), 
        sources=('train2014/frame', 'train2014/img', 'train2014/capt'), 
        load_in_memory=True)
    
    num_examples_on_gpu = batch_size * batches_on_gpu
    num_examples = data.num_examples // num_examples_on_gpu
    num_examples *= num_examples_on_gpu
    
    data.example_iteration_scheme = ShuffledScheme(num_examples, num_examples_on_gpu)
    data.default_transformers = uint8_pixels_to_floatX(('train2014/frame', 'train2014/img'))
        
    gan = GAN(num_examples_on_gpu, voc_size)
    if params_file:
        gan.load_params(params_file)
    
    logging.info('Starting training: num_epochs={}, max_example_stream_iter={}, batches_on_gpu={}, unroll={}, data_file={}'.format(num_epochs, max_example_stream_iter, batches_on_gpu, unroll, data_file))
   
    data_stream = data.apply_default_transformers(
        data.get_example_stream())
    
    for epoch in range(num_epochs):
        
        start_time = time.time()
        # In each epoch, we do a full pass over the training data:
        train_D_loss = 0
        train_G_loss = 0
        
        processed_D = 0
        processed_G = 0
        
        data_stream.next_epoch()
        for it_num, (frames, imgs, caps) in enumerate(data_stream.get_epoch_iterator()):
            
            logging.debug('{}: Loading {} examples to GPU'.format(it_num+1, len(imgs)))
            
            gan.frames_var.set_value(frames.transpose(0,3,1,2))
            gan.img_var.set_value(imgs.transpose(0,3,1,2))
            gan.noise_var.set_value(lasagne.utils.floatX(np.random.randn(len(imgs),100)))
            gan.caps_var.set_value(sparse_floatX(caps))
             
            len_imgs = len(imgs)
            actual_batches = int(math.ceil(len_imgs / batch_size))
             
            logging.debug('Done loading {} examples to GPU'.format(len_imgs))
            logging.debug('frames.shape={}, imgs.shape={}, caps.shape={}'.format(frames.shape, imgs.shape, caps.shape))
             
            fake_i = 0
             
            for i in range(actual_batches):
                 
                first = i * batch_size
                last = min(first + batch_size, len_imgs) 
                 
                """
                Train discriminator on real images right away, but delay training on fake ones
                Accumulate all minibatches, and then train discriminator and generator on fake images
                """ 
#                 logging.debug('real, first={}, last={}'.format(first, last))
                train_D_loss += gan.train_D_real(first, last) * (last-first)
                processed_D += last-first
                   
                if (i+1) % unroll == 0:
                    # train generator with accumulated batches
                    while fake_i <= i: 
                        fake_first = fake_i * batch_size
                        fake_last = min(fake_first + batch_size, len_imgs)
#                         logging.debug('fake, first={}, last={}'.format(fake_first, fake_last))
                        train_D_loss += gan.train_D_fake(fake_first, fake_last) * (fake_last-fake_first)
                        processed_D += fake_last-fake_first
                        train_G_loss += gan.train_G(fake_first, fake_last) * (fake_last-fake_first)
                        processed_G += fake_last-fake_first
                        fake_i+=1
                         
            if max_example_stream_iter and it_num+1 >= max_example_stream_iter:
                break
#         
#       
        logging.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logging.info("  avg training loss (D/G):\t\t{}".format(np.array([train_D_loss / processed_D, train_G_loss / processed_G])))
  
        # Be on a safe side - if the job is killed, it is better to preserve at least something
        if (epoch+1) % out_freq == 0 or epoch == num_epochs - 1:
            gan.save_params('{}.{}'.format(out_model, epoch + 1))
              
#    predict(gan, data_fp, 'val2014', 10)
        

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Trains a DCGAN on COCO using Lasagne')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-n', '--num_epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-u', '--unroll', type=int, default=1, help='unroll (num mini-batches) (default=None)')
    parser.add_argument('-p', '--params_dir', type=str, help='directory with parameters files (npz format)')
    parser.add_argument('-b', '--max_example_stream_iter', type=int, help='the total max number_of_batches_to_train *  batches_on_gpu (defailt: None, meaning train using all examples). If provided, it will be multiplied by delay_g_training')
    parser.add_argument('-s', '--batch_size', type=int, default=128, help='the number of examples per batch')
    parser.add_argument('-o', '--out_model', type=str, default='../models/GAN', help='otput model')
    parser.add_argument('-f', '--output_freq', type=int, default=1, help='frequency of output (default: 1, which means each epoch)')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    parser.add_argument('-g', '--batches_on_gpu', type=int, default=10, help='number of mini-batches to load simultaneously on GPU')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
 
    train(args.data_file, out_model=args.out_model, out_freq=args.output_freq, 
          voc_size=11172,  num_epochs=args.num_epochs, batch_size=args.batch_size,
          params_file=args.params_dir, batches_on_gpu=args.batches_on_gpu, unroll=args.unroll, 
          max_example_stream_iter=args.max_example_stream_iter)
