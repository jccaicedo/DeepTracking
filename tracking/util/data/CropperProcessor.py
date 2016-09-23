# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:14:51 2016

@author: MindLAB
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class CropperProcessor(Processor):
    
    def __init__(self, cropper, positionModel):
        self.cropper = cropper
        self.theta = None
        self.positionModel = positionModel
        
        
    def preprocess(self, frame, position):
        frame, position, self.theta = self.cropper.crop(frame, position, position)

        return frame, position

    
    def postprocess(self, position):
        
        # Transforming the positions
        position = self.positionModel.transform(self.theta, position)

        return position