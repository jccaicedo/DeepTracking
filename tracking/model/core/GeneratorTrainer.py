# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:57:53 2016

@author: MindLab
"""

import logging
from Trainer import Trainer

class GeneratorTrainer(Trainer):

    """
    Create a Trainer for a tracker.

    @type  tracker: tracking.model.Tracker
    @param tracker: The tracker to be trained
    """
    def __init__(self, tracker, generator, processor, validator):
        self.tracker = tracker
        self.generator = generator
        self.processor = processor
        self.validator = validator
    
    
    """
    Train a tracker.

    @type  milestones: tracking.model.Tracker
    @param milestones: The tracker to be trained
    @type  batches: tracking.model.Tracker
    @param batches: The tracker to be trained
    @type  batchSize: tracking.model.Tracker
    @param batchSize: The tracker to be trained
    @type  lnr: tracking.model.Tracker
    @param lnr: The tracker to be trained
    @type  lnrdy: tracking.model.Tracker
    @param lnrdy: The tracker to be trained
    """
    def train(self, milestones, batches, batchSize, lnr, lnrdy):
        for milst in range(milestones):
            for batch in range(batches):
                frame, gtPosition = self.generator.getBatch(batchSize)
                frame, prepGtPosition = self.processor.preprocess(frame, gtPosition)                
                loss = self.tracker.fit(frame, prepGtPosition, lnr)              
                self.validator.validateBatch(self.tracker, frame, prepGtPosition)
                
                logging.info("Batch Loss: Milestone = %d, batch = %d, loss = %f", milst, batch, loss)
            
            # Updating the learning rate
            lnr *= lnrdy
                
            self.validator.validateEpoch(self.tracker)        