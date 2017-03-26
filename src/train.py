import theano
import lasagne

import numpy as np

from DCGAN import DCGAN
from utils import iterate_minibatches


import h5py

import sys

import time

def train(num_epochs, initial_eta, data_fp, split, params_file = None):
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    batch_size=100
    
    gan = DCGAN()
    
    # Load the dataset
  
    if not params_file == None:
        gan.load_params(params_file)
    
    # Finally, launch the training loop.
    print("Starting training...")
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_discr_loss = 0
        train_gen_loss = 0
        train_batches = 0
        start_time = time.time()
        
        for batch in iterate_minibatches(data_fp, split, batch_size, shuffle=False):
            _, _, y_var = batch
            y_var = lasagne.utils.floatX(y_var) / 255

            discr_loss, gen_loss = gan.train(y_var)
            train_discr_loss += discr_loss
            train_gen_loss += gen_loss
                                           
            train_batches += 1
            
            break
      

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (D/G):\t\t{}".format(np.array([train_discr_loss, train_gen_loss]) / train_batches))
            
    gan.save_params('../models/DCGAN')
            
def main(data_file, num_epochs=100, initial_eta=2e-4, params_file = None):
    
    print("Loading data...")
    with h5py.File(data_file,'r') as hf:
        train(num_epochs, initial_eta, hf, 'train2014', params_file)


if __name__ == '__main__':
    
    if ('--help' in sys.argv) or ('-h' in sys.argv) or (len(sys.argv) < 2):
        print("Trains a DCGAN on COCO using Lasagne.")
        print("Usage: %s [DATA_FILE] [EPOCHS] [MODEL_PARAMS_FILES]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 100)")
    else:
        kwargs = {}
        kwargs['data_file'] = str(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['params_file'] = str(sys.argv[3])
            
        main(**kwargs)
