import gensim
import os

import cPickle as pkl

import logging

from tokenizer import Sentences

def train_w2v(caption_dict, n_out, out_file):

    logging.info('Training word2vec...')    
    sentences = Sentences(caption_dict)
  
    model = gensim.models.Word2Vec(sentences, size=n_out, min_count=3)
    
    logging.info('Finished training word2vec')
    logging.info('Vocabulary size: {}'.format(len(model.wv.vocab)))
          
    model.save('{}.{}.model'.format(out_file, n_out))
    
    return model.wv
    
def main(n_out = 512):
    
    data_path = '../resources/inpainting'
    with open(os.path.join(data_path ,'dict_key_imgID_value_caps_train_and_valid.pkl')) as fd:
        caption_dict = pkl.load(fd)
    
    word_vectors = train_w2v(caption_dict, n_out, '../models/word2vec')
 
  #  plotWords()
     
    # shows the similar words
    print (word_vectors.most_similar('bathroom'))
    
    print (word_vectors.most_similar('bicycle'))
    
    print (word_vectors.most_similar('beach'))
    
    print (word_vectors.most_similar('sitting'))
    
    print (word_vectors.most_similar('standing'))
    
    print (word_vectors.most_similar('petersburg'))
    
    print (word_vectors.most_similar('tramway'))
    
    word_vectors.save_word2vec_format('../models/word2vec.%d.wv' % n_out,
                                      '../models/word2vec.%d.voc' % n_out)
    
    wv = gensim.models.KeyedVectors.load_word2vec_format('../models/word2vec.%d.wv' % n_out,
                                      '../models/word2vec.%d.voc' % n_out)
    
    
    print wv.syn0.shape
     
    # shows the learnt embedding
   # print (model['bicycl'])
if __name__=='__main__':
    
    logging.basicConfig(level=logging.DEBUG)
    main()       