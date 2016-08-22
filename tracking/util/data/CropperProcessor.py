# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:14:51 2016

@author: MindLAB
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class CropperProcessor(Processor):
    
    def __init__(self, cropper, processor, positionModel):
        self.cropper = cropper
        self.processor = processor
        self.theta = None
        self.positionModel = positionModel
        
        
    def preprocess(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        if position.shape[1] > 1:
            cropPosition = NP.roll(position, 1, axis=1) # Shift the time
            cropPosition[:, 0, :] = cropPosition[:, 1, :] # First frame is ground truth
        else:
            cropPosition = NP.copy(position)
        frame, position, self.theta = self.cropper.crop(frame, cropPosition, position)

        return frame, position

    
    def postprocess(self, frame, position):
        batchSize, seqLength, channels, height, width = frame.shape
        
        # Transforming the positions
        position = self.positionModel.transform(self.theta, position)
        
        frame, position = self.processor.postprocess(frame, position)

        return frame, position