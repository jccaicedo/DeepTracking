# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:38:33 2016

@author: MindLab
"""

from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from tracking.model.core.Regressor import Regressor

class Regressor(Regressor):
    
    def __init__(self, layers):
        self.build(layers)
        
    
    def build(self, layers):
        model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        self.model = TimeDistributed(model)
        
    
    def getModel(self):
        
        return self.model
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable
            
            
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        pass