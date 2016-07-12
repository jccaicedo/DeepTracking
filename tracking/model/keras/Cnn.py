# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:11:10 2016

@author: MindLab
"""

from keras.models import Sequential
from tracking.model.core.Cnn import Cnn

class Cnn(Cnn):
    
    def __init__(self, layers):
        self.buildModel(layers)
        
    
    def buildModel(self, layers):
        model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        self.model = model
        
    
    def getOutputDim(self, inDims):
        outDims = self.model.output_shape
        
        return outDims
        
        
    def getModel(self):
        return self.model
        
        
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable