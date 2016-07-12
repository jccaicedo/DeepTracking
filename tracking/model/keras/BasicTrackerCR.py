# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 11:08:16 2016

@author: MindLab
"""

from tracking.model.core.Tracker import Tracker
from keras.models import Sequential
from keras.optimizers import RMSprop

class BasicTrackerCR(Tracker):
    
    def __init__(self, batchSize, layers, lnr):
        self.batchSize = batchSize
        self.buildModel(layers, lnr)
        

    def fit(self, frame, position, lnr):
        batchSize, seqLength, channels, height, width = frame.shape
        targetDim = position.shape[2]
        frame = frame.reshape((batchSize * seqLength, channels, height, width))
        position = position.reshape((batchSize * seqLength, targetDim))
        history = self.model.fit(frame, position, batch_size=self.batchSize, nb_epoch=1, verbose=0)
        
        loss = history.history["loss"][0]
        
        return loss
    
    
    def forward(self, frame, initPosition):
        batchSize, seqLength, channels, height, width = frame.shape
        targetDim = initPosition.shape[1]
        frame = frame.reshape((batchSize * seqLength, channels, height, width))
        position = self.model.predict(frame, batch_size=self.batchSize, verbose=0)
        
        position = position.reshape((batchSize, seqLength, targetDim))
        
        return position
        
    
    def buildModel(self, layers, lnr):
        model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        optimizer=RMSprop(lr=lnr, rho=0.9, epsilon=1e-08),
        model.compile(optimizer=optimizer, loss='mse')
        self.model = model