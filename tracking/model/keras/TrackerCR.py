# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:40:10 2016

@author: MindLab
"""

from keras.models import Model
from keras.layers import Input
from tracking.model.keras.Tracker import Tracker

class TrackerCR(Tracker):
    
    def __init__(self, cnn, regressor, batchSize, frameDims, optimizer):
        self.frameDims = frameDims
        self.cnn = cnn
        self.regressor = regressor
        self.buildModel(optimizer)
    
    
    def fit(self, frame, position, lnr):
        batchSize, seqLength, channels, height, width = frame.shape
        targetDim = position.shape[2]
        frame = frame.reshape((batchSize * seqLength, channels, height, width))
        position = position.reshape((batchSize * seqLength, targetDim))
        loss = self.model.train_on_batch(frame, position)
        
        return loss
    
    
    def forward(self, frame, initPosition):
        batchSize, seqLength, channels, height, width = frame.shape
        targetDim = initPosition.shape[1]
        frame = frame.reshape((batchSize * seqLength, channels, height, width))
        position = self.model.predict_on_batch(frame)
        
        position = position.reshape((batchSize, seqLength, targetDim))
        
        return position
    
    
    def buildModel(self, optimizer):
        inLayer = Input(shape=self.frameDims, name="input")
        feat = self.cnn.getModel()(inLayer)
        position = self.regressor.getModel()(feat)
        self.model = Model(input=inLayer, output=position)
        self.model.compile(optimizer=optimizer, loss='mse')    