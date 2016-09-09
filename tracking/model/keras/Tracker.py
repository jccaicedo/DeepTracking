# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:09:36 2016

@author: MindLab
"""

import logging
import numpy as NP
from keras.callbacks import Callback
from keras.models import Model
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker):
    
    def __init__(self, input, modules, builder, optimizer, loss, processor, timeSize):
        self.input = input
        self.modules = modules
        self.builder = builder
        self.optimizer = optimizer
        self.loss = loss
        self.processor = processor
        self.timeSize = timeSize
        self.build()
    
    
    def fit(self, input, position, lnr):
        loss = self.model.train_on_batch(input, position)
        
        return loss
        
    
    def predict(self, input):
        position = self.model.predict_on_batch(input)
    
        return position
    
    
    def forward(self, frame, initPosition):
        batchSize = frame.shape[0]
        seqLength = frame.shape[1]
        targetDim = initPosition.shape[1]
        iters = seqLength / self.timeSize + (seqLength % self.timeSize > 0)
        position = NP.empty((batchSize, 0, targetDim))
        predPosition = initPosition
        
        for i in range(iters):
            start = self.timeSize * (i)
            end = self.timeSize * (i + 1)
            pFrame = frame[:, start:end, ...]
            predPosition = self.step(pFrame, predPosition)
            position = NP.append(position, predPosition, axis=1)
    
        return position
    
    
    def build(self):
        output = self.builder.build(self.input, self.modules)
        model = Model(input=self.input, output=output)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model = model
        
        
    def train(self, generator, epochs, batches, batchSize, validator):
        history = LossHistory(validator, self)
        spe = batches * batchSize
        self.model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=spe, verbose=0, callbacks=[history])
        
    
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful, batchSize):
        
        for module in self.modules:
            module.setStateful(stateful, batchSize)
            
        self.build()
        
    
    def reset(self):
        self.model.reset_states()
        
        
    def getWeights(self):
        
        return self.model.get_weights()
        
    
    def setWeights(self, weights):
        self.model.set_weights(weights)
        
        
    def step(self, frame, position):
        input = self.processor.before(frame, position)
        position = self.model.predict_on_batch(input)
        position = self.processor.after(position)
        
        return position
        
        
class LossHistory(Callback):
    
    def __init__(self, validator, tracker):
        self.validator = validator
        self.tracker = tracker


    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", 0, batch, loss)
        
        
    def on_epoch_end(self, epoch, logs={}):
        self.validator.validateEpoch(self.tracker)