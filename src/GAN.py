#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import theano
from theano import tensor as T
import lasagne
import numpy as np

import logging

from model import Model 
from collections import OrderedDict

class Discriminator(Model):
    
    def __init__(self, img_var=None, capt_var=None):
        
        Model.__init__(self, "Discriminator")
        
        layers = []
    
        custom_rectify = lasagne.nonlinearities.LeakyRectify(0.1)
        
        self.logger.info('-----------build_discriminator-------------')
        
        self.in_caps = lasagne.layers.InputLayer(
                shape=(None, 11188), 
                name='InputLayer_Capture', 
                input_var=capt_var)
        
        self.in_img = lasagne.layers.InputLayer(
                shape=(None, 3, 64, 64),
                name='InputLayer_Img',
                input_var=img_var)
        
        layers.append(self.in_img)
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
         
        layers.append(
            lasagne.layers.Conv2DLayer(
                layers[-1],
                name='Conv2DLayer1',
                num_filters=64, filter_size=5, stride=2, pad=2,
                nonlinearity=custom_rectify,
                W=lasagne.init.HeNormal(gain='relu')))
              
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer2', 
                    num_filters=128, filter_size=5, stride=2, pad=2,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
          
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
         
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer3', 
                    num_filters=256, filter_size=5, stride=2, pad=2,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))

        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer4', 
                    num_filters=512, filter_size=5, stride=2, pad=2,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
                
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
          
        layers.append(
            lasagne.layers.FlattenLayer(
                layers[-1], 
                name='FlattenLayer'))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.ConcatLayer(
                [layers[-1], self.in_caps],
                name='ConcatLayer'))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(
                    layers[-1], 
                    name='DenseLayer', 
                    num_units=1,
                    nonlinearity=lasagne.nonlinearities.sigmoid,
                    W=lasagne.init.HeNormal(gain=1.0))))
        
    
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        self.nn = layers[-1]   
        
#         print ('params', len(lasagne.layers.get_all_param_values(self.nn)))

class Generator(Model):

    def __init__(self, noise_var=None, border_var=None, caps_var=None):
        
        Model.__init__(self, "Generator")
    
        custom_rectify = lasagne.nonlinearities.rectify #LeakyRectify(0.1)
    
        self.logger.info('-----------build_generator-------------')
        
        self.in_noise = lasagne.layers.InputLayer(
                shape=(None, 100), 
                name='InputLayer_Noise', 
                input_var=noise_var)
                
        self.in_border = lasagne.layers.InputLayer(
                shape=(None, 3, 64, 64), 
                name='InputLayer_Border', 
                input_var=border_var)
        
        self.in_caps = lasagne.layers.InputLayer(
                shape=(None, 11188), 
                name='InputLayer_Capture', 
                input_var=caps_var)
        
        layers = []
        layers.append(self.in_border)
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
              
        layers.append(
            lasagne.layers.Conv2DLayer(
                layers[-1],
                name='Conv2DLayer1',
                num_filters=64, filter_size=5, stride=2, pad=2,
                nonlinearity=custom_rectify,
                W=lasagne.init.HeNormal(gain='relu')))
              
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
#         
#         layers.append(
#             lasagne.layers.MaxPool2DLayer(
#                  layers[-1],
#                  name='MaxPoll2DLayer1',
#                  pool_size=2))
#         
#         self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
#         
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer2', 
                    num_filters=128, filter_size=5, stride=2, pad=2,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
          
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
#         layers.append(
#             lasagne.layers.MaxPool2DLayer(
#                  layers[-1],
#                  name='MaxPoll2DLayer2',
#                  pool_size=2))
#         
#         self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
         
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer3', 
                    num_filters=256, filter_size=5, stride=2, pad=2,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))

        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer4', 
                    num_filters=512, filter_size=5, stride=2, pad=2,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
                 
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
          
        layers.append(
            lasagne.layers.FlattenLayer(
                layers[-1], 
                name='FlattenLayer'))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.ConcatLayer(
                [layers[-1], self.in_noise, self.in_caps],
                name='ConcatLayer'))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
    
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(
                    layers[-1], 
                    name='DenseLayer1', 
                    num_units=512*4*4,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.GlorotUniform('relu'))))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.reshape(
                layers[-1], 
                name='ReshapeLayer', 
                shape=(-1, 512, 4, 4)))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.Upscale2DLayer(
                layers[-1],
                name='Upscale2DLayer4',
                scale_factor=2,
                mode='repeat'))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer4', 
                    nonlinearity=custom_rectify,
                    num_filters=256, filter_size=5, stride=1, pad=2,
                    W=lasagne.init.HeNormal(gain='relu'))))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.Upscale2DLayer(
                layers[-1],
                name='Upscale2DLayer3',
                scale_factor=2,
                mode='repeat'))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    layers[-1], 
                    name='Conv2DLayer3', 
                    nonlinearity=custom_rectify,
                    num_filters=128, filter_size=5, stride=1, pad=2,
                    W=lasagne.init.HeNormal(gain='relu'))))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.Upscale2DLayer(
                layers[-1],
                name='Upscale2DLayer2',
                scale_factor=2,
                mode='repeat'))
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
#         layers.append(
#             lasagne.layers.batch_norm(
#                 lasagne.layers.Conv2DLayer(
#                     layers[-1], 
#                     name='Conv2DLayer2', 
#                     nonlinearity=custom_rectify,
#                     num_filters=64, filter_size=5, stride=1, pad=2,
#                     W=lasagne.init.HeNormal(gain='relu'))))
#          
#         self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
#         
#         layers.append(
#             lasagne.layers.Upscale2DLayer(
#                 layers[-1],
#                 name='Upscale2DLayer1',
#                 scale_factor=2,
#                 mode='repeat'))
#       
#        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.Conv2DLayer(
                layers[-1], 
                name='Conv2DLayer1',
                nonlinearity=lasagne.nonlinearities.sigmoid, 
                num_filters=3, filter_size=5, stride=1, pad=2,
                W=lasagne.init.HeNormal(gain=1.0)))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.PadLayer(
                layers[-1], 
                name='PadLayer',
                width=16))
         
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        layers.append(
            lasagne.layers.ElemwiseSumLayer(
                [layers[-1], self.in_border],
                name='InpaintLayer')
            )
        
        self.logger.debug('{}, {}'.format(layers[-1].name, layers[-1].output_shape))
        
        self.nn = layers[-1]
        
#         print ('params', len(lasagne.layers.get_all_param_values(self.nn)))
        
class GAN(object):
    
    def __init__(self):
        
        self.logger = logging.getLogger(__name__)
        
        noise = T.matrix('noise') 
        border = T.tensor4('border')
        border = border.dimshuffle((0, 3, 1, 2))
        images = T.tensor4('images')
        images = images.dimshuffle((0, 3, 1, 2))
        
        caps = T.matrix('caps')
    
        self.generator = Generator(noise, border, caps)
        self.discriminator = Discriminator(images, caps)
        
        self.logger.info("Compiling Theano functions...")
        
#         noise_vec, border_img = lasagne.layers.get_output(
#             [self.generator.in_noise, self.generator.in_border])

        inp_G = OrderedDict()
        inp_G[self.generator.in_caps] = caps
        inp_G[self.generator.in_border] = border
        inp_G[self.generator.in_noise] = noise
        
        img_fake = lasagne.layers.get_output(self.generator.nn, inputs=inp_G)
    #    img_fake_determ = lasagne.layers.get_output(self.generator.nn, inputs=noise, deterministic=True)
    
        inp_D_real = OrderedDict()
        inp_D_real[self.discriminator.in_img] = images
        inp_D_real[self.discriminator.in_caps] = caps
        
        
        # Create expression for passing real data through the discriminator
        probs_real = lasagne.layers.get_output(self.discriminator.nn, inputs=inp_D_real)
    #    probs_real_determ = lasagne.layers.get_output(self.discriminator.nn, inputs=images, deterministic=True)
    
        inp_D_fake = OrderedDict()
        inp_D_fake[self.discriminator.in_img] = img_fake
        inp_D_fake[self.discriminator.in_caps] = caps
         
        # Create expression for passing fake data through the discriminator
        probs_fake = lasagne.layers.get_output(self.discriminator.nn, inputs=inp_D_fake)
    #    probs_fake_determ = lasagne.layers.get_output(self.discriminator.nn, inputs=img_fake_determ, deterministic=True)
        
         
        # Create loss expressions
        loss_G = lasagne.objectives.binary_crossentropy(probs_fake, 0.9).mean()               
        loss_D_real = lasagne.objectives.binary_crossentropy(probs_real, 0.9).mean()
        loss_D_fake = lasagne.objectives.binary_crossentropy(probs_fake, 0.0).mean()
        loss_D = loss_D_real + loss_D_fake
        
                 
        # Create update expressions for training
        params_G = lasagne.layers.get_all_params(self.generator.nn, trainable=True)
        params_D = lasagne.layers.get_all_params(self.discriminator.nn, trainable=True)
         
        #   eta = theano.shared(lasagne.utils.floatX(initial_eta))
        updates_G = lasagne.updates.adam(loss_G, params_G, learning_rate=0.0002, beta1=0.5)
        updates_D = lasagne.updates.adam(loss_D, params_D, learning_rate=0.0002, beta1=0.5)
        updates_D_real = lasagne.updates.adam(loss_D_real, params_D, learning_rate=0.0002, beta1=0.5)
        updates_D_fake = lasagne.updates.adam(loss_D_fake, params_D, learning_rate=0.0002, beta1=0.5)
        
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_D = theano.function(
                [images,caps,noise,border],
                outputs=loss_D,
                updates=updates_D
                )
        
        self.train_D_real = theano.function(
                [images,caps],
                outputs=loss_D_real,
                updates=updates_D_real
                )
        
        self.train_D_fake = theano.function(
                [caps,noise,border],
                outputs=loss_D_fake,
                updates=updates_D_fake
                )
        
        self.train_G = theano.function(
                [caps,noise,border],
                outputs=loss_G,
                updates=updates_G
                )
#         
#         self.test_G_inputs = theano.function(
#                 [noise,border],
#                 outputs=[noise_vec,border_img]
#                 )
#     
        # Compile another function generating some data
        self.predict_fake = theano.function(
            [noise,border,caps],
            outputs=[img_fake, probs_fake]
        ) 
    
        self.predict_real = theano.function(
            [images,caps],
            outputs=[probs_real]
        ) 
    
    def train(self, caps_var, border_var, image_var, noise_var):
        
        return (self.train_D(image_var, caps_var, noise_var, border_var), 
                self.train_G(caps_var, noise_var, border_var))
        
    def train_real(self, image_var, caps_var):
        return self.train_D_real(image_var, caps_var)
    
    def train_fake(self, noise_var, border_var, caps_var):
        return (self.train_D_fake(caps_var, noise_var, border_var),
                self.train_G(caps_var, noise_var, border_var))
    
    def predict(self, border_var, caps_var, noise_var):
        return self.predict_fake(noise_var, border_var, caps_var)  
    
    def load_params(self, file_name):
        
        self.generator.load_params(file_name)  
        self.discriminator.load_params(file_name) 
            
    def save_params(self, file_name):
        
        self.generator.save_params(file_name)  
        self.discriminator.save_params(file_name)  
    
    