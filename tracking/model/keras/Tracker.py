# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:09:36 2016

@author: MindLab
"""

import logging
from keras.callbacks import Callback
from keras.layers import Input
from keras.models import Model
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker):
    
    def __init__(self, models, optimizer, loss, inputShape):
        self.inputShape = inputShape
        self.buildModel(models, optimizer, loss, inputShape)
    
    
    def fit(self, frame, position, lnr):
        loss = self.model.train_on_batch(frame, position)
        
        return loss
    
    
    def forward(self, frame, initPosition):
        #last_output, outputs, states = K.rnn(step, X, initial_states=[])
        position = self.model.predict_on_batch(frame)
        
        return position
    
    
    def buildModel(self, models, optimizer, loss, inputShape):
        inLayer = Input(shape=inputShape)
        outLayer = inLayer
        
        for model in models:
            outLayer = model(outLayer)
        
        model = Model(input=inLayer, output=outLayer)
        model.compile(optimizer=optimizer, loss=loss)
        self.model = model
        
        
    def train(self, generator, epochs, batches, batchSize, validator):
        history = LossHistory(validator, self)
        spe = batches * batchSize
        self.model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=spe, verbose=0, callbacks=[history])
        
        
    def reset(self):
        self.model.reset_states()
        
        
    def getWeights(self):
        
        return self.model.get_weights()
        
    
    def setWeights(self, weights):
        self.model.set_weights(weights)
        
        
class LossHistory(Callback):
    
    def __init__(self, validator, tracker):
        self.validator = validator
        self.tracker = tracker


    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", 0, batch, loss)
        
        
    def on_epoch_end(self, epoch, logs={}):
        self.validator.validateEpoch(self.tracker)