# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:06:51 2016

@author: MindLAB
"""

import theano.tensor as THT
from keras.layers import Merge
from tracking.util.theano.Data import Data

class SquareAttention():
    
    def __init__(self, layers, alpha):
        if type(layers) is not list or len(layers) != 2:
            raise Exception("SquareAttention must be called on a list of two layers. Got: " + str(layers))
        
        mode = lambda X: SquareAttention.call(X, alpha)
        
        self.model = Merge(layers=layers, mode=mode, output_shape=SquareAttention.getOutputShape)


    def getModel(self):
        return self.model


    @staticmethod
    def call(X, alpha):
        if type(X) is not list or len(X) != 2:
            raise Exception("SpatialTransformer must be called on a list of two tensors. Got: " + str(X))
           
        frame, position  = X[0], X[1]
        (batchSize, chans, height, width) = frame.shape
        rX = Data.linspace(-1.0, 1.0, width)
        rY = Data.linspace(-1.0, 1.0, height)
        FX = THT.gt(rX, position[:,0].dimshuffle(0,'x')) * THT.le(rX, position[:,2].dimshuffle(0,'x'))
        FY = THT.gt(rY, position[:,1].dimshuffle(0,'x')) * THT.le(rY, position[:,3].dimshuffle(0,'x'))
        m = FY.dimshuffle(0, 1, 'x') * FX.dimshuffle(0, 'x', 1)
        m = m + alpha - THT.gt(m, 0.) * alpha
        
        return frame * m.dimshuffle(0,'x',1,2)


    @staticmethod
    def getOutputShape(inputShapes):
        frameShape = inputShapes[0]
        
        return frameShape        