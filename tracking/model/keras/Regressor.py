# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:38:33 2016

@author: MindLab
"""

from keras.models import Sequential
from tracking.model.core.Regressor import Regressor

class Regressor(Regressor):
    
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