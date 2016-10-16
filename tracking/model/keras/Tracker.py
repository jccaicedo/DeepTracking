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
    
    def __init__(self, input=None, modules=None, builder=None, optimizer=None, loss=None, processor=None, timeSize=None):
        self.input = input
        self.modules = modules
        self.builder = builder
        self.optimizer = optimizer
        self.loss = loss
        self.processor = processor
        self.timeSize = timeSize
    
    
    def fit(self, input, position, lnr):
        loss = self.model.train_on_batch(input, position)
        
        return loss
        
    
    def predict(self, input):
        position = self.model.predict_on_batch(input)
    
        return position
    
    
    def forward(self, input, initPosition):
        batchSize = input[0].shape[0]
        seqLength = input[0].shape[1]
        targetDim = initPosition.shape[1]
        iters = seqLength / self.timeSize + (seqLength % self.timeSize > 0)
        position = NP.empty((batchSize, 0, targetDim))
        predPosition = initPosition
        
        for i in range(iters):
            start = self.timeSize * (i)
            end = self.timeSize * (i + 1)
            pInput = [inP[:, start:end, ...] for inP in input]
            predPosition = self.step(pInput, predPosition)
            position = NP.append(position, predPosition, axis=1)
    
        return position
    
    
    def build(self):
        output = self.builder.build(self.input, self.modules)
        model = Model(input=self.input, output=output)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model = model
        print self.model.summary()
        
        
    def train(self, generator, epochs, batches, batchSize, validator):
        self.generator = generator
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
        
        for name, module in self.modules.items():
            module.setStateful(stateful, batchSize)
            
        self.build()
        
    
    def reset(self):
        self.model.reset_states()
        
        
    def getWeights(self):
        
        return self.model.get_weights()
        
    
    def setWeights(self, weights):
        self.model.set_weights(weights)
        
        
    def step(self, input, position):
        input = self.processor.before(input, position)
        position = self.model.predict_on_batch(input)
        position = self.processor.after(position)
        
        return position
        
        
class LossHistory(Callback):
    
    def __init__(self, validator, tracker):
        self.validator = validator
        self.tracker = tracker
        self.epoch = 0


    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        logging.info("Batch Loss: Epoch = %d, batch = %d, loss = %f", self.epoch, batch, loss)
        '''from keras import backend as K
        get_activations = K.function([self.model.layers[0].input, self.model.layers[1].input],[self.model.layers[2].get_output_at(0)])
        import cPickle as pickle
        sample = pickle.load(open("/home/ubuntu/tracking/data/debug_batch.pkl","r"))
        activations = get_activations(sample) 
        with open("/home/ubuntu/tracking/data/debug_act.pkl", "wb") as out:
            pickle.dump(activations, out)'''
        from keras import backend as K
        print self.model.layers
        activations = K.function([self.model.layers[1].input, self.model.layers[0].input],[self.model.layers[3].get_output_at(1)])

        import cPickle as pickle
        sample = pickle.load(open("/home/ubuntu/tracking/data/debug_batch.pkl","r"))
        activations = activations(sample)
        with open("/home/ubuntu/tracking/data/debug_trans.pkl", "wb") as out:
            pickle.dump(activations, out)

        
    def on_epoch_end(self, epoch, logs={}):
        self.validator.validateEpoch(self.tracker)
        self.epoch += 1
        print sample[0].shape
        print sample[0].shape
