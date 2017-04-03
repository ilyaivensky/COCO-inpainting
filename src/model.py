#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lasagne
import numpy as np
import logging

class Model(object):
    
    def __init__(self, name):

        self.nn = None
        self.name = name
        self.logger = logging.getLogger(self.name)

    def save_params(self, file_name):
        
        fname = "%s.%s.npz" % (file_name, self.name)
        params = lasagne.layers.get_all_param_values(self.nn) 
        self.logger.info("Saving network params to {}, total {}".format(fname, len(params)))
        np.savez(fname, *params)
       
    def load_params(self, file_name):
        
        fname = "%s.%s.npz" % (file_name, self.name)
        self.logger.info("Loading network params from {}".format(fname))
    
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.nn, param_values)
        