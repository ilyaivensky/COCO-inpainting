import sys

import theano
from lasagne import utils as lu
from numpy import random as rnd 

from DCGAN import DCGAN
from utils import iterate_minibatches
from utils import show_samples

import h5py
import argparse

def predict(data_fp, split, params_file):
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    rnd.seed(87537)
    
    batch_size=50
    
    gan = DCGAN()     
    gan.load_params(params_file)
    
    print("Starting predicting...")
   
    predict_loss = 0
    nr_batches = 0
    
    for batch in iterate_minibatches(data_fp, split, batch_size, shuffle=False):
        _, _, y_var = batch
        y_var = lu.floatX(y_var) / 255
        
        nb_examples = 10
        samples, loss = gan.predict(y_var, nb_examples)
        show_samples(y_var[:nb_examples], samples)
        predict_loss += loss
        nr_batches += 1
        
        break

    print("  prediction loss:\t\t{}".format(predict_loss / nr_batches))

def main(data_file, params_file):  
    
    print("Loading data...")
    with h5py.File(data_file,'r') as hf:
        predict(hf, 'val2014', params_file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Tests predictions of trained DCGAN on COCO val2014 dataset')
    parser.add_argument('data_file', help='h5 file with prepocessed dataset')
    parser.add_argument('params_dir', type=str, help='directory with parameters files (npz format)')
    args = parser.parse_args()
    
    main(args.data_file, args.params_dir)