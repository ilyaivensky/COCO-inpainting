#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.sparse as sparse
from lasagne.layers import DenseLayer
from lasagne import nonlinearities, init

from scipy.sparse import csr_matrix

class DenseLayerSparseInput(DenseLayer):
    
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):    
        
        super(DenseLayerSparseInput, self).__init__(incoming, num_units, W,
                 b, nonlinearity,
                 num_leading_axes=1, **kwargs)
        
    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        activation = sparse.structured_dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)
    
def sparse_floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return csr_matrix(arr, dtype=theano.config.floatX)