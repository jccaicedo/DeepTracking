# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:09:36 2016

@author: MindLab
"""

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker):
    
    def __init__(self, cnn, rnn, regressor, batchSize, frameDims):
        self.frameDims = frameDims
        self.batchSize = batchSize
        self.cnn = cnn
        self.rnn = rnn
        self.regressor = regressor
        self.buildModel()
    
    
    def fit(self, frame, position, lnr):
        history = self.model.fit(frame, position, batch_size=self.batchSize, nb_epoch=1, verbose=0)
        
        loss = history.history["loss"][0]
        
        return loss
    
    
    def forward(self, frame, initPosition):
        position = self.model.predict(frame, batch_size=self.batchSize, verbose=0)
        
        return position
    
    
    def buildModel(self):
        model = Sequential()
        cnn = TimeDistributed(self.cnn.getModel(), input_shape=(None, ) + self.frameDims)
        rnn = self.rnn.getModel()
        reg = TimeDistributed(self.regressor.getModel())
        model.add(cnn)
        model.add(rnn)
        model.add(reg)
        model.compile(optimizer='rmsprop', loss='mse')
        
        self.model = model
        
        
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful):
        self.rnn.setStateful(stateful)
        
        self.buildModel()