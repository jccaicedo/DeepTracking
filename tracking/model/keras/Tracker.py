# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:09:36 2016

@author: MindLab
"""

import logging
from keras.callbacks import Callback
from keras.models import Model
from tracking.model.core.Tracker import Tracker

class Tracker(Tracker):
    
    def __init__(self, input, modules, optimizer, loss, inputShape):
        self.input = input
        self.modules = modules
        self.optimizer = optimizer
        self.loss = loss
        self.inputShape = inputShape
        self.build()
    
    
    def fit(self, input, position, lnr):
        loss = self.model.train_on_batch(input, position)
        
        return loss
    
    
    def forward(self, input, initPosition):
        #pframe = frame[:, :1, ...]
        #last_output, outputs, states = K.rnn(self.step, frame, initial_states=[initPosition])
        #return outputs
        position = self.model.predict_on_batch(input)
        
        return position
    
    
    def build(self):
        output = self.input
        
        for module in self.modules:
            output = module.getModel()(output)
        
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
        
        
    def step(self, frame, states):
        position = states[0]
        frame, position = self.step(frame, position)
        position = self.model.predict_on_batch(frame)
        frame, position = self.stepPos(frame, position)
        
        return position, [position]
        
        
class LossHistory(Callback):
    
    def __init__(self, validator, tracker):
        self.validator = validator
        self.tracker = tracker


    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", 0, batch, loss)
        
        
    def on_epoch_end(self, epoch, logs={}):
        self.validator.validateEpoch(self.tracker)