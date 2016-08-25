# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:24:21 2016

@author: MindLAB
"""

from keras import backend as K
from keras.layers import Merge
from theano import tensor as T

class Transformer():
    
    def __init__(self, layers):
        if type(layers) is not list or len(layers) != 2:
            raise Exception("SpatialTransformer must be called on a list of two layers. Got: " + str(layers))
        
        self.model = Merge(layers=layers, mode=Transformer.call, output_shape=Transformer.getOutputShape)


    def getModel(self):
        return self.model


    @staticmethod
    def call(X):
        if type(X) is not list or len(X) != 2:
            raise Exception("SpatialTransformer must be called on a list of two tensors. Got: " + str(X))
                            
        theta, position  = X[0], X[1]
        theta = theta.reshape((-1, 2, 3))
        output = Transformer.transform(theta, position)

        return output


    @staticmethod
    def getOutputShape(inputShapes):
        positionShape = inputShapes[1]
        
        return positionShape
        
        
    @staticmethod
    def transform(theta, position):
        (batchSize, targetDim) = K.shape(position)
        
        # Reshaping the positions
        position = K.reshape(K.flatten(position), (batchSize, 2, -1))
        position = K.permute_dimensions(position, (0, 2, 1))
        position = K.concatenate([position, T.ones((batchSize, 1, K.shape(position)[2]))], axis=1)
        
        # Applying the transformation
        position = K.batch_dot(theta, position)[:, :2, :]
        
        # Reshaping the result
        position = K.permute_dimensions(position, (0, 2, 1))
        position = K.reshape(position, (batchSize, targetDim))
        
        return position