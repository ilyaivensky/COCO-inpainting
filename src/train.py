import theano
import lasagne

import numpy as np

from DCGAN import DCGAN
from utils import iterate_minibatches


import h5py

import argparse
import time

def train(num_epochs, num_batches, initial_eta, data_fp, split, delay_g_training = 1, params_file = None):
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    np.random.seed(87537)
    
    batch_size=100
    
    gan = DCGAN()
    
    # Load the dataset
  
    if not params_file is None:
        gan.load_params(params_file)
        
    if (not num_batches is None):
        num_batches *= delay_g_training
        
    # Finally, launch the training loop.
    print("Starting training...")
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_D_loss = 0
        train_G_loss = 0
        train_batches = 0
        
        start_time = time.time()
        
        accumulated_batches = [None] * delay_g_training
        
        for batch in iterate_minibatches(data_fp, split, batch_size, shuffle=False):
            
            train_batches += 1
            
            _, _, y_var = batch
            y_var = lasagne.utils.floatX(y_var) / 255

            if delay_g_training > 1:
                """
                Train discriminator right away, but delay generator training.
                Accumulate all minibatches, and then provide them to generator
                """ 
                noise_var = lasagne.utils.floatX(np.random.randint(low=0, high=256, size=(len(y_var),100)))
                train_D_loss += gan.train_D(y_var.transpose(0,3,1,2), noise_var)
                
                acc_idx = train_batches % delay_g_training
                accumulated_batches[acc_idx] = batch
            
                if  acc_idx == 0:
                    # train generator with accumulated batches
                    for acc_batch in accumulated_batches:
                        _, _, y_var = acc_batch
                        y_var = lasagne.utils.floatX(y_var) / 255
                        noise_var = lasagne.utils.floatX(np.random.randint(slow=0, high=256, ize=(len(y_var),100)))
                        train_G_loss += gan.train_G(noise_var)
            else:
                loss_D, loss_G = gan.train(y_var)
                train_D_loss += loss_D
                train_G_loss += loss_G
                
            if (not num_batches is None) and (train_batches == num_batches):
                break
      

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (D/G):\t\t{}".format(np.array([train_D_loss, train_G_loss]) / train_batches))
            
    gan.save_params('../models/DCGAN')
            
def main(data_file, num_epochs=100, num_batches=None, initial_eta=2e-4, delay_g_training=1, params_file=None):
    
    print("Loading data...")
    with h5py.File(data_file,'r') as hf:
        train(num_epochs, num_batches, initial_eta, hf, 'train2014', delay_g_training, params_file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Trains a DCGAN on COCO using Lasagne')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('-n', '--num_epochs', type=int, default=100, help='number of epochs (default: 100)')
    parser.add_argument('-d', '--delay_g_training', type=int, default=1, help='delay (num mini-batches) in generator training (default=1)')
    parser.add_argument('-p', '--params_dir', type=str, help='directory with parameters files (npz format)')
    parser.add_argument('-b', '--num_batches', type=int, help='the max number of batches to train (defailt: None, meaning train all batches). If provided, it will be multiplied by delay_g_training')
    
    args = parser.parse_args()
 
    main(args.data_file, num_epochs=args.num_epochs, 
         params_file=args.params_dir, delay_g_training=args.delay_g_training, num_batches=args.num_batches)
