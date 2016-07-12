# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:55:29 2016

@author: MindLab
"""

import logging
import numpy as NP

class Validator(object):

    def __init__(self, frame, position, batchSize, measure):        
        self.frame = frame
        self.position = position
        self.batchSize = batchSize
        self.measure = measure
        
    
    def validateEpoch(self, tracker):
        valSetSize = self.frame.shape[0] 
        seqLength = self.frame.shape[1]
        targetDim = self.position.shape[2]
        iters = valSetSize / self.batchSize + (valSetSize % self.batchSize > 0)
        predPosition = NP.empty((0, seqLength, targetDim))
        
        for i in range(iters):
            start = self.batchSize * (i)
            end = self.batchSize * (i + 1)
            frame = self.frame[start:end, ...]
            position = self.position[start:end, 0, ...]
            tracker.reset()
            batchPredPosition = tracker.forward(frame, position)
            predPosition = NP.append(predPosition, batchPredPosition, axis=0)        
        
        measureValue = self.measure.calculate(self.position, predPosition).mean()
        logging.info("Validation Epoch: %s = %f", self.measure.name, measureValue)
        
    
    def validateBatch(self, tracker, frame, position):
        tracker.reset()
        predPosition = tracker.forward(frame, position[:, 0, :])
        measureValue = self.measure.calculate(position, predPosition).mean()
        logging.info("Validation Batch: %s = %f", self.measure.name, measureValue)
        

    def setValidationSet(self, frame, position):
        self.frame = frame
        self.position = position
        
        
    def test(self, tracker, seqs, processor):
        result = {}

        for name, frame, position in seqs:
            frame = NP.expand_dims(frame, axis=0)
            position = NP.expand_dims(position, axis=0)
            prepF, prepP = processor.preprocess(frame, position)
            tracker.reset()
            predP = tracker.forward(prepF, prepP[:, 0, :])
            _, postPredP = processor.postprocess(prepF, predP)
            measure = self.measure.calculate(position, postPredP)
    
            result[name] = measure
    
    
        return result