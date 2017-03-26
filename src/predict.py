import sys

import theano
from lasagne import utils as lu

from DCGAN import DCGAN
from utils import iterate_minibatches
from utils import show_samples

import h5py

def predict(data_fp, split, params_file):
    
    theano.config.floatX = 'float32'
    theano.exception_verbosity='high'
    
    batch_size=100
    
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
    
    if ('--help' in sys.argv) or ('-h' in sys.argv) or (len(sys.argv) < 3):
        print("Predicts on COCO using Lasagne.")
        print("Usage: %s [h5 DATA_FILE] [MODEL_PARAMS_FILES]" % sys.argv[0])
        print()
    
    else:
        kwargs = {}
        kwargs['data_file'] = str(sys.argv[1])
        kwargs['params_file'] = str(sys.argv[2])          

        main(**kwargs)