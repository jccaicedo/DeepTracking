# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:06:45 2016

@author: MindLab
"""

from keras.models import Sequential
from tracking.model.core.Rnn import Rnn

class Rnn(Rnn):
    
    def __init__(self, layers):
        self.buildModel(layers)
        
    
    def buildModel(self, layers):
        model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        self.model = model
        
    
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
    def setStateful(self, stateful):
        for layer in self.model.layers:
            layer.stateful = stateful