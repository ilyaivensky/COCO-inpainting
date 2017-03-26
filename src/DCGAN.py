import theano
from theano import tensor as T

import lasagne
    
import numpy as np

class Model(object):
    
    def __init__(self, name):

        self.nn = None
        self.name = name
        
    def save_params(self, file_name):
        
        fname = "%s.%s.npz" % (file_name, self.name)
        params = lasagne.layers.get_all_param_values(self.nn) 
        print (self.name, ': saving network params to', fname, 'total', len(params))
        np.savez(fname, *params)
       
    def load_params(self, file_name):
        
        fname = "%s.%s.npz" % (file_name, self.name)
        print (self.name, ': loading network params from', fname)
    
        with np.load(fname) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.nn, param_values)
        
        
class Discriminator(Model):
    
    def __init__(self, input_var=None):
        
        Model.__init__(self, "Discriminator")
        self.layers = []
    
        custom_rectify = lasagne.nonlinearities.LeakyRectify(0.1)
        
        print ('-----------build_discriminator-------------')
        self.layers.append(
            lasagne.layers.InputLayer(
                shape=(None, 3, 64, 64),
                name='InputLayer',
                input_var=input_var))
        
        x_shape = self.layers[-1].shape
        
        print (self.layers[-1].name, x_shape)
         
        self.layers.append(
            lasagne.layers.Conv2DLayer(
                self.layers[-1],
                name='Conv2DLayer1',
                num_filters=64, filter_size=4, stride=2, pad=1,
                nonlinearity=custom_rectify,
                W=lasagne.init.HeNormal(gain='relu')))
              
        print (self.layers[-1].name, self.layers[-1].output_shape)
      
#         self.layers.append(
#             lasagne.layers.MaxPool2DLayer(
#                 self.layers[-1], 
#                 name='MaxPool2DLayer1', pool_size=(2, 2)))
#         
#           
#         x_shape = self.layers[-1].get_output_shape_for(x_shape)
#         print (self.layers[-1].name, x_shape)
#           
        
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    self.layers[-1], 
                    name='Conv2DLayer2', 
                    num_filters=128, filter_size=4, stride=2, pad=1,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
          
        print (self.layers[-1].name, self.layers[-1].output_shape)
          
#         self.layers.append(
#             lasagne.layers.MaxPool2DLayer(
#                 self.layers[-1], 
#                 name='MaxPool2DLayer2', 
#                 pool_size=(2, 2)))
#         
#         
#         x_shape = self.layers[-1].get_output_shape_for(x_shape)
#         print (self.layers[-1].name, x_shape)
#         
#         
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    self.layers[-1], 
                    name='Conv2DLayer3', 
                    num_filters=256, filter_size=4, stride=2, pad=1,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
        
        print (self.layers[-1].name, self.layers[-1].output_shape)
          
#         self.layers.append(
#             lasagne.layers.MaxPool2DLayer(
#                 self.layers[-1], 
#                 name='MaxPool2DLayer3', 
#                 pool_size=(2, 2)))
#         
#         x_shape = self.layers[-1].get_output_shape_for(x_shape)
#         print (self.layers[-1].name, x_shape)     

        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Conv2DLayer(
                    self.layers[-1], 
                    name='Conv2DLayer4', 
                    num_filters=512, filter_size=4, stride=2, pad=1,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.HeNormal(gain='relu'))))
                
        print (self.layers[-1].name, self.layers[-1].output_shape)
          
#         self.layers.append(
#             lasagne.layers.MaxPool2DLayer(
#                 self.layers[-1], 
#                 name='MaxPool2DLayer3', 
#                 pool_size=(2, 2)))
#         
#         x_shape = self.layers[-1].get_output_shape_for(x_shape)
#         print (self.layers[-1].name, x_shape)      
    
        self.layers.append(
            lasagne.layers.FlattenLayer(
                self.layers[-1], 
                name='FlattenLayer'))
        
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(
                    self.layers[-1], 
                    name='DenseLayer', 
                    num_units=1,
                    nonlinearity=lasagne.nonlinearities.sigmoid,
                    W=lasagne.init.HeNormal(gain=1.0))))
        
    
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.nn = self.layers[-1]    
        
#         print ('params', len(lasagne.layers.get_all_param_values(self.nn)))

class Generator(Model):

    def __init__(self, input_var=None):
        
        Model.__init__(self, "Generator")
        
        self.layers = []
        
        custom_rectify = lasagne.nonlinearities.rectify #LeakyRectify(0.1)
    
        print ('-----------build_generator-------------')
        
        self.layers.append(
            lasagne.layers.InputLayer(
                shape=(None, 100), 
                name='InputLayer', 
                input_var=input_var))
        
        print (self.layers[-1].name, self.layers[-1].output_shape) 
    
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(
                    self.layers[-1], 
                    name='DenseLayer1', 
                    num_units=512*4*4,
                    nonlinearity=custom_rectify,
                    W=lasagne.init.GlorotUniform('relu'))))
        
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.layers.append(
            lasagne.layers.reshape(
                self.layers[-1], 
                name='ReshapeLayer', 
                shape=(-1, 512, 4, 4)))
         
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Deconv2DLayer(
                    self.layers[-1], 
                    name='Deconv2DLayer4', 
                    nonlinearity=custom_rectify,
                    num_filters=256, filter_size=4, stride=2, crop=1,
                    W=lasagne.init.HeNormal(gain='relu'))))
         
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Deconv2DLayer(
                    self.layers[-1], 
                    name='Deconv2DLayer3', 
                    nonlinearity=custom_rectify,
                    num_filters=128, filter_size=4, stride=2, crop=1,
                    W=lasagne.init.HeNormal(gain='relu'))))
         
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.layers.append(
            lasagne.layers.batch_norm(
                lasagne.layers.Deconv2DLayer(
                    self.layers[-1], 
                    name='Deconv2DLayer2', 
                    nonlinearity=custom_rectify,
                    num_filters=64, filter_size=4, stride=2, crop=1,
                    W=lasagne.init.HeNormal(gain='relu'))))
         
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.layers.append(
            lasagne.layers.Deconv2DLayer(
                self.layers[-1], 
                name='Deconv2DLayer1',
                nonlinearity=lasagne.nonlinearities.sigmoid, 
                num_filters=3, filter_size=4, stride=2, crop=1,
                W=lasagne.init.HeNormal(gain=1.0)))
         
        print (self.layers[-1].name, self.layers[-1].output_shape)
        
        self.nn = self.layers[-1]
#         print ('params', len(lasagne.layers.get_all_param_values(self.nn)))
        
class DCGAN(object):
    
    def __init__(self):
        
        z = T.matrix('z') 
        y = T.tensor4('y')
        y = y.dimshuffle((0, 3, 1, 2))
    
        self.generator = Generator(z)
        self.discriminator = Discriminator(y)
        
        print("Compiling Theano functions...")
        
        img_fake = lasagne.layers.get_output(self.generator.nn, inputs=z)
        img_fake_determ = lasagne.layers.get_output(self.generator.nn, inputs=z, deterministic=True)
        
        # Create expression for passing real data through the discriminator
        probs_real = lasagne.layers.get_output(self.discriminator.nn, inputs=y)
        probs_real_determ = lasagne.layers.get_output(self.discriminator.nn, inputs=y, deterministic=True)
         
        # Create expression for passing fake data through the discriminator
        probs_fake = lasagne.layers.get_output(self.discriminator.nn, inputs=img_fake)
        probs_fake_determ = lasagne.layers.get_output(self.discriminator.nn, inputs=img_fake_determ, deterministic=True)
        
         
        # Create loss expressions
        generator_loss = lasagne.objectives.binary_crossentropy(probs_fake, 1).mean()
        discriminator_loss = (lasagne.objectives.binary_crossentropy(probs_real, 1)
               + lasagne.objectives.binary_crossentropy(probs_fake, 0)).mean()
                 
                 
        # Create update expressions for training
        generator_params = lasagne.layers.get_all_params(self.generator.nn, trainable=True)
        discriminator_params = lasagne.layers.get_all_params(self.discriminator.nn, trainable=True)
         
        #   eta = theano.shared(lasagne.utils.floatX(initial_eta))
        generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=0.001, beta1=0.9)
        discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=0005, beta1=0.6)
        
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_d = theano.function(
                [y,z],
                outputs=discriminator_loss,
                updates=discriminator_updates
                )
        
        self.train_g = theano.function(
                [z],
                outputs=generator_loss,
                updates=generator_updates
                )
    
        # Compile another function generating some data
        self.predict_fake = theano.function(
            [z],
            outputs=[img_fake_determ, probs_fake_determ]
        ) 
    
        self.predict_real = theano.function(
            [y],
            outputs=[probs_real_determ]
        ) 
    
    def train(self, target):
        
        noise = lasagne.utils.floatX(np.random.uniform(size=(len(target),100)))
        return (self.train_d(target.transpose(0,3,1,2), noise), self.train_g(noise))
    
    def predict(self, target, nb_samples):
        
        noise = lasagne.utils.floatX(np.random.uniform(size=(nb_samples,100)))
        return self.predict_fake(noise)
   
    
    def load_params(self, file_name):
        
        self.generator.save_params(file_name)  
        self.discriminator.save_params(file_name) 
            
    def save_params(self, file_name):
        
        self.generator.save_params(file_name)  
        self.discriminator.save_params(file_name)  
    
    