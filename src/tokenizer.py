#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# 
from nltk.tokenize import RegexpTokenizer
#from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Tokenizer(object):
    
    def __init__(self):
        
        self.en_stop = set(stopwords.words('english'))
        self.en_stop.update(['a','the'])
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()
        
    def iterate_sentences(self, caption):
        
        for sentence in caption:
            tokens = self.tokenizer.tokenize(sentence.lower())
            words = [t for t in tokens if not t in self.en_stop]
            
#                     # remove numbers
#                     number_tokens = [re.sub(r'[\d]', ' ', t) for t in words]
#                     number_tokens = ' '.join(number_tokens).split()
#  
#                     # stem tokens
#                     stemms = [stemmer.stem(t) for t in number_tokens]

            yield words
            
    def iterate_words(self, caption):
        
        for sentence in self.iterate_sentences(caption):
            for word in sentence:
                yield word
            
    
class Sentences(object):
    
    def __init__(self, caption_dict):
        self.caption_dict = caption_dict
        self.tokenizer = Tokenizer()
 
    def __iter__(self):
        
        for index,doc in self.caption_dict.iteritems():
            
            if index.find('train2014') != -1: 
                for w in self.tokenizer.iterate_sentences(doc):
                    yield w
