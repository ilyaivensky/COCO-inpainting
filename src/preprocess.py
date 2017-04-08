import os
import subprocess
import glob

import cPickle as pkl
from PIL import Image

import numpy as np
from scipy import sparse

import h5py

from _collections import defaultdict

import logging

from word2vec import train_w2v
from tokenizer import Tokenizer



def preprocess_split(data_path, caption_dict, wv):

    """
    """
    
    tokenizer = Tokenizer()
    imgs = glob.glob(os.path.join(data_path,"*.jpg"))
    
    x = np.empty((len(imgs), 64, 64, 3), dtype=np.uint8) #observed images w/black-outs
    y = np.empty((len(imgs), 64, 64, 3), dtype=np.uint8) #full imgs
    ids = np.empty(len(imgs), dtype=np.uint32)
    cap_features = sparse.lil_matrix((len(imgs), len(wv.vocab)), dtype=np.uint32)
    
    logging.info('len(imgs) ={}'.format(len(imgs)))
    
    i = 0
    for img_path in imgs:
        
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]
        

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        # Skip B/W images
        if len(img_array.shape) != 3:
            continue
        
        y[i] = np.copy(img_array)
        x[i] = np.copy(img_array)
        x[i,center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
        ids[i] = int(os.path.basename(img_path)[-10:-4])
        
        capture = caption_dict[cap_id]
        
        caprure_idxs = defaultdict()
        
        for token in tokenizer.iterate_words(capture):
            
            try:
                caprure_idxs[wv.vocab[token].index] =+ 1 
            except KeyError:
                logging.debug('Image {}: unknown word "{}"'.format(img_path, token))
                pass
       
        if not caprure_idxs:
            logging.info('Image {}: {}'.format(img_path, capture))
        for k,v in caprure_idxs.items():
            cap_features[i,k] = v
                  
        i += 1
         
    # Truncate the empty tail (remember - we have allocated space for all images but ignored B/W ones)
    return ids[:i], x[:i], y[:i], cap_features.tocsr()[:i]

    
def main():
    
    """
    Preprocess data (training and validation) and store it in HDF5 file
    """
    
    data_path = '../resources/inpainting'
    
    with open(os.path.join(data_path ,'dict_key_imgID_value_caps_train_and_valid.pkl')) as fd:
        caption_dict = pkl.load(fd)
        
#     with open(os.path.join(data_path ,'worddict.pkl')) as wfd:
#         vocabulary = pkl.load(wfd)
#         
#     print len(vocabulary)
#     for item in vocabulary.items():
#         print item
#         
#         break
#     return
        
    wv = train_w2v(caption_dict, 512, '../models/word2vec')
    
    #make savedir
    preprocessed_path = '../preprocessed'
    
    if not os.path.exists(preprocessed_path):
        command0 = "mkdir -p" +" "+ preprocessed_path
        subprocess.check_call(command0.split())
        
    with h5py.File(os.path.join(preprocessed_path,'inpainting.h5'),'w') as hf:
        
        
        for split in ['train2014','val2014']:
            
            logging.info('processing {}'.format(split))
            split_path = os.path.join(data_path, split)
            ids,x,y,capt = preprocess_split(split_path, caption_dict, wv)
            logging.info('{} examples: {}'.format(split, len(x)))
            
            grp = hf.create_group(split)
            
            ids_data = grp.create_dataset("id", (ids.shape), dtype=np.uint32)
            ids_data[...] = ids
            
            x_data = grp.create_dataset("frame", (x.shape), dtype=np.uint8)
            x_data[...] = x
            y_data = grp.create_dataset("img", (y.shape), dtype=np.uint8)
            y_data[...] = y
            
            capt_grp = grp.create_group('capt')
            capt_grp.attrs['shape0'] = capt.shape[0]
            capt_grp.attrs['shape1'] = capt.shape[1]
            
            capt_data = capt_grp.create_dataset("data", (capt.data.shape), dtype=np.uint32)
            capt_data[...] = capt.data
            capt_indices = capt_grp.create_dataset("indices", (capt.indices.shape), dtype=np.uint32)
            capt_indices[...] = capt.indices
            capt_indptr = capt_grp.create_dataset("indptr", (capt.indptr.shape), dtype=np.uint32)
            capt_indptr[...] = capt.indptr

if __name__=='__main__':
    
    logging.basicConfig(level=logging.INFO)
    main()       