#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division


import theano

import numpy as np

from GAN import GAN
from utils import setup_logging, generate_z

import argparse
import time

import logging

from dataset import H5PYSparseDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
#from fuel.transformers import Transformer
from fuel.transformers.defaults import uint8_pixels_to_floatX
import fuel
#from PIL import Image

from sparse_matrix_utils import sparse_floatX

#from evaluate_GAN import evaluate_GAN

def train(data_file, out_model, out_freq, voc_size, num_epochs, batch_size, batches_on_gpu, delay_g, max_example_stream_iter=None, 
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
    
    # num_examples_on_gpu has to be fixed, 
    # therefore the number of examples we use for training should divide by num_examples_on_gpu
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
        train_D_real_loss = 0
        train_D_fake_loss = 0
        train_G_loss = 0
        
        last_train_D_real_loss = None
        last_train_D_fake_loss = None
        last_train_G_loss = None
        
        processed_D_real = 0
        processed_D_fake = 0
        processed_G = 0
        
        data_stream.next_epoch()
        for it_num, (frames, imgs, caps) in enumerate(data_stream.get_epoch_iterator()):
            
            gan.frames_var.set_value(frames.transpose(0,3,1,2))
            gan.img_var.set_value(imgs.transpose(0,3,1,2))
            gan.noise_var.set_value(generate_z((len(imgs),gan.generator.noise_shape)))
            gan.caps_var.set_value(sparse_floatX(caps))
            
            logging.debug('{}: Loaded {} examples to GPU'.format(it_num+1, len(imgs)))
           
            gen_i = 0  
            for i in range(batches_on_gpu):
                 
                first = i * batch_size
                last = first + batch_size
                 
                """
                Train discriminator right away, but delay training of generator
                """ 
#                 logging.debug('real, first={}, last={}'.format(first, last))
                last_train_D_real_loss = gan.train_D_real(first, last) * (last-first)
                train_D_real_loss += last_train_D_real_loss
                last_train_D_real_loss /= (last-first)
                processed_D_real += last-first
                
                last_train_D_fake_loss = gan.train_D_fake(first, last) * (last-first)
                train_D_fake_loss += last_train_D_fake_loss
                last_train_D_fake_loss /= (last-first)
                processed_D_fake += last-first
                   
                if epoch >= delay_g and (i+1) % unroll == 0:
                    # train generator with accumulated batches
                    while gen_i <= i: 
                        gen_first = gen_i * batch_size
                        gen_last = gen_first + batch_size
#                         logging.debug('fake, first={}, last={}'.format(gen_first, gen_last))
                        
                        last_train_G_loss = gan.train_G(gen_first, gen_last) * (gen_last-gen_first)
                        train_G_loss += last_train_G_loss
                        last_train_G_loss /= (gen_last-gen_first)
                        processed_G += gen_last-gen_first
                        gen_i+=1
                         
            if max_example_stream_iter and it_num+1 >= max_example_stream_iter:
                break
#       
        if processed_D_real:
            train_D_real_loss /= processed_D_real 
        else:
            train_D_real_loss = None
            
        if processed_D_fake:
            train_D_fake_loss /= processed_D_fake 
        else:
            train_D_fake_loss = None
            
        if processed_G:
            train_G_loss /= processed_G
        else:
            train_G_loss = None
        
#       
        logging.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logging.info("  epoch avg training loss (DR/DF/G):\t\t{}".format(
            np.array([train_D_real_loss, train_D_fake_loss, train_G_loss])))
        logging.info("  last batch training loss (DR/DF/G):\t\t{}".format(
            np.array([last_train_D_real_loss, last_train_D_fake_loss, last_train_G_loss])))
  
        # Be on a safe side - if the job is killed, it is better to preserve at least something
        if (epoch+1) % out_freq == 0 or epoch == num_epochs - 1:
            gan.save_params('{}.{:03d}'.format(out_model, epoch + 1))
              
#    evaluate_GAN(gan, data_fp, 'val2014', 10)
        

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Trains a DCGAN on COCO using Lasagne')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-n', '--num_epochs', type=int, default=1000, help='number of epochs (default: 1000)')
#     parser.add_argument('-u', '--unroll', type=int, default=5, help='unroll (num mini-batches) (default=None)')
    parser.add_argument('-p', '--params_dir', type=str, help='directory with parameters files (npz format)')
    parser.add_argument('-b', '--max_example_stream_iter', type=int, help='the total max number_of_batches_to_train *  batches_on_gpu (defailt: None, meaning train using all examples). If provided, it will be multiplied by delay_g_training')
    parser.add_argument('-s', '--batch_size', type=int, default=128, help='the number of examples per batch')
    parser.add_argument('-o', '--out_model', type=str, default='../models/GAN', help='otput model')
    parser.add_argument('-f', '--output_freq', type=int, default=1, help='frequency of output (default: 1, which means each epoch)')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    parser.add_argument('-g', '--batches_on_gpu', type=int, default=10, help='number of mini-batches to load simultaneously on GPU')
    parser.add_argument('-d', '--delay_g', type=int, default=5, help='number of epoch to delay training generator')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
 
    train(args.data_file, out_model=args.out_model, out_freq=args.output_freq, 
          voc_size=11172,  num_epochs=args.num_epochs, batch_size=args.batch_size,
          params_file=args.params_dir, batches_on_gpu=args.batches_on_gpu, delay_g=args.delay_g, 
          unroll=args.batches_on_gpu, 
          max_example_stream_iter=args.max_example_stream_iter)
