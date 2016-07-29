# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:09:36 2016

@author: MindLab
"""

import logging
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers.wrappers import TimeDistributed
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker):
    
    def __init__(self, cnn, rnn, regressor, frameDims, optimizer):
        self.frameDims = frameDims
        self.cnn = cnn
        self.rnn = rnn
        self.regressor = regressor
        self.optimizer = optimizer
        self.buildModel(optimizer)
    
    
    def fit(self, frame, position, lnr):
        loss = self.model.train_on_batch(frame, position)
        
        return loss
    
    
    def forward(self, frame, initPosition):
        position = self.model.predict_on_batch(frame)
        
        return position
    
    
    def buildModel(self, optimizer):
        model = Sequential()
        cnn = TimeDistributed(self.cnn.getModel(), input_shape=(None, ) + self.frameDims)
        rnn = self.rnn.getModel()
        reg = TimeDistributed(self.regressor.getModel())
        model.add(cnn)
        model.add(rnn)
        model.add(reg)
        model.compile(optimizer=self.optimizer, loss='mse')
        
        self.model = model
        
        
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        self.rnn.setStateful(stateful, batchSize)
        
        self.buildModel(self.optimizer)
        
        
    def train(self, generator, epochs, batches, batchSize, validator):
        history = LossHistory(validator, self)
        spe = batches * batchSize
        self.model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=spe, verbose=0, callbacks=[history])
        
        
    def reset(self):
        self.model.reset_states()
        
        
class LossHistory(Callback):
    
    def __init__(self, validator, tracker):
        self.validator = validator
        self.tracker = tracker


    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", 0, batch, loss)
        
        
    def on_epoch_end(self, epoch, logs={}):
        self.validator.validateEpoch(self.tracker)