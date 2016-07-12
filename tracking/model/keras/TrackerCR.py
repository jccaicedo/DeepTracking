# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:40:10 2016

@author: MindLab
"""

from keras.models import Model
from keras.layers import Input
from tracking.model.core.Tracker import Tracker

class TrackerCR(Tracker):
    
    def __init__(self, cnn, regressor, batchSize, imgDims):
        self.batchSize = batchSize
        self.cnn = cnn
        self.regressor = regressor
        self.buildModel(imgDims)
    
    
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
    
    
    def buildModel(self, imgDims):
        inLayer = Input(shape=imgDims, name="input")
        feat = self.cnn.getModel()(inLayer)
        position = self.regressor.getModel()(feat)
        self.model = Model(input=inLayer, output=position)
        self.model.compile(optimizer='rmsprop', loss='mse')    