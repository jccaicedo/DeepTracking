# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 01:05:00 2016

@author: MindLAB
"""

from keras import backend as K
from keras.engine.topology import Layer


class Transformer(Layer):
    
    def __init__(self, downsampleFactor=1, **kwargs):
        self.downsampleFactor = downsampleFactor
        super(Transformer, self).__init__(**kwargs)


    def call(self, X, mask=None):
        position, theta = X
        batchSize = input.shape[0]
        theta = theta.reshape((batchSize, 2, 3))
        output = self.transform(theta, input)

        return output


    def get_output_shape_for(self, inputShape):
        
        return inputShape
        
    
    # theta.shape = (batchSize, 3, 3)
    @staticmethod
    def transform(self, theta, position):
        batchSize, targetDim = position.shape
        
        # Reshaping the positions
        position = K.flatten(position).reshape((batchSize, 2, -1))
        position = K.permute_dimensions(position, (0, 2, 1))
        position = K.concatenate([position, K.ones((batchSize, 1, position.shape[2]))], axis=1)
        
        # Applying the transformation
        position = K.batched_dot(theta, position)[:, :2, :]
        
        # Reshaping the result
        position = K.permute_dimensions(position, (0, 2, 1))
        position = position.reshape((batchSize, targetDim))
        
        return position