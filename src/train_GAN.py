#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import theano
import lasagne

import numpy as np

from GAN import GAN
from utils import DataIterator, setup_logging

import h5py

import argparse
import time

import logging

#from predict import predict

def train(out_model, num_epochs, num_batches, initial_eta, data_fp, split, unroll = 1, params_file = None):
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    np.random.seed(87537)
    
    batch_size=128
    
    gan = GAN()
    
    # Load the dataset
   
    if params_file is not None:
        gan.load_params(params_file)
         
    if (num_batches is not None):
        num_batches *= unroll
        
    iterator = DataIterator(data_fp, split)
         
    # Finally, launch the training loop.
    logging.info('Starting training: num_epochs={}, num_batches={}, unroll={}, data_fp={}'.format(num_epochs, num_batches, unroll, data_fp))
     
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_D_loss = 0
        train_G_loss = 0
        train_batches = 0
         
        start_time = time.time()
         
        if unroll is not None:
            accumulated_batches = [None] * unroll
         
        for batch in iterator.iterate_minibatches(batch_size, shuffle=False):
             
            train_batches += 1
             
            _, x, y, caps = batch
            
            x_var = (lasagne.utils.floatX(x) / 255).transpose(0,3,1,2)
            y_var = (lasagne.utils.floatX(y) / 255).transpose(0,3,1,2)
            
            noise_var = lasagne.utils.floatX(np.random.randn(len(y_var),100))
            
            caps_var = lasagne.utils.floatX(caps)
 
            if unroll is not None:
                """
                Train discriminator on real images right away, but delay training on fake ones
                Accumulate all minibatches, and then train discriminator and generator on fake images
                """ 
                train_D_loss += gan.train_real(y_var, caps_var)
                 
                acc_idx = train_batches % unroll
                accumulated_batches[acc_idx] = (x_var, noise_var, caps_var)
              
                if acc_idx == 0:
                    # train generator with accumulated batches
                    for acc_batch in accumulated_batches: 
                        x_var, noise_var, caps_var = acc_batch
                        loss_D, loss_G = gan.train_fake(noise_var, x_var, caps_var)
                        train_D_loss += loss_D
                        train_G_loss += loss_G
                        
            else:
                loss_D, loss_G = gan.train(caps_var, x_var, y_var, noise_var)
                train_D_loss += loss_D
                train_G_loss += loss_G
                 
            if (num_batches is not None) and (train_batches == num_batches):
                break
       
 
        # Then we print the results for this epoch:
        logging.info("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        logging.info("  training loss (D/G):\t\t{}".format(np.array([train_D_loss, train_G_loss]) / train_batches))

        # Be on a safe side - if the job is killed, it is better to preserve at least something
        if epoch % 10 == 9 or epoch == num_epochs - 1:
            gan.save_params('{}.{}'.format(out_model, epoch + 1))
             
#    predict(gan, data_fp, 'val2014', 10)
            
def main(data_file, out_model, num_epochs=100, num_batches=None, initial_eta=2e-4, unroll=1, params_file=None, log_file=None):
    
    logger = logging.getLogger(__name__)
    logger.info('Loading data from {}...'.format(data_file))
    
    with h5py.File(data_file,'r') as hf:
        train(out_model, num_epochs, num_batches, initial_eta, hf, 'train2014', unroll, params_file)


if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Trains a DCGAN on COCO using Lasagne')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='number of epochs (default: 100)')
    parser.add_argument('-u', '--unroll', type=int, default=None, help='unroll (num mini-batches) (default=None)')
    parser.add_argument('-p', '--params_dir', type=str, help='directory with parameters files (npz format)')
    parser.add_argument('-b', '--num_batches', type=int, help='the max number of batches to train (defailt: None, meaning train all batches). If provided, it will be multiplied by delay_g_training')
    parser.add_argument('-o', '--out_model', type=str, default='../models.DCGAN', help='otput model')
    parser.add_argument('-l', '--log_file', type=str, default='logging.yaml', help='file name with logging configuration')
    
    args = parser.parse_args()
    
    setup_logging(default_path=args.log_file)
 
    main(args.data_file, num_epochs=args.num_epochs, 
         params_file=args.params_dir, unroll=args.unroll, out_model=args.out_model, num_batches=args.num_batches)
