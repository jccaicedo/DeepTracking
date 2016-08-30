# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:55:29 2016

@author: MindLAB
"""

import logging
import numpy as NP

class Validator(object):

    def __init__(self, input, position, batchSize, measure):        
        self.input = input
        self.position = position
        self.batchSize = batchSize
        self.measure = measure
        
    
    def validateEpoch(self, tracker):
        valSetSize = self.input.shape[0] 
        seqLength = self.input.shape[1]
        targetDim = self.position.shape[2]
        iters = valSetSize / self.batchSize + (valSetSize % self.batchSize > 0)
        predPosition = NP.empty((0, seqLength, targetDim))
        
        for i in range(iters):
            start = self.batchSize * (i)
            end = self.batchSize * (i + 1)
            input = self.input[start:end, ...]
            position = self.position[start:end, 0, ...]
            tracker.reset()
            batchPredPosition = tracker.forward(input, position)
            predPosition = NP.append(predPosition, batchPredPosition, axis=0)        
        
        measureValue = self.measure.calculate(self.position, predPosition).mean()
        logging.info("Validation Epoch: %s = %f", self.measure.name, measureValue)
        
    
    def validateBatch(self, tracker, input, position):
        tracker.reset()
        predPosition = tracker.forward(input, position[:, 0, :])
        measureValue = self.measure.calculate(position, predPosition).mean()
        logging.info("Validation Batch: %s = %f", self.measure.name, measureValue)
        

    def setValidationSet(self, input, position):
        self.input = input
        self.position = position
        
        
    def test(self, tracker, seqs):
        result = {}

        for name, input, position in seqs:
            tracker.reset()
            predP = tracker.forward(input, position[:, 0, :])
            measure = self.measure.calculate(position, predP)
    
            result[name] = measure
    
        return result