# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:02:56 2016

@author: MindLab
"""

import numpy as NP
from PIL import ImageDraw
from tracking.model.core.PositionModel import PositionModel

class CentroidPM(PositionModel):
    
    def __init__(self, height, width):
        super(CentroidPM, self).__init__(2)
        self.height = height
        self.width = width
        
    
    def getTargetDim(self):
        
        return 2
        
        
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def fromTwoCorners(self, position):
        x1 = position[:, :, 0]
        y1 = position[:, :, 1]
        x2 = position[:, :, 2]
        y2 = position[:, :, 3]
        
        xCenter = x1 + x2 / 2.0
        yCenter = y1 + y2 / 2.0
        
        newPosition = NP.zeros((position.shape[0], position.shape[1], 2))
        
        newPosition[:, :, 0] = xCenter
        newPosition[:, :, 1] = yCenter
        
        return newPosition
    
    
    # position.shape = (batchSize, seqLength, targetDim(xC, yC))
    def toTwoCorners(self, position):
        xCenter = position[:, :, 0]
        yCenter = position[:, :, 1]
        
        newPosition = NP.zeros((position.shape[0], position.shape[1], 4))
        
        newPosition[:, :, 0] = xCenter - self.width / 2.0
        newPosition[:, :, 1] = yCenter - self.height / 2.0
        newPosition[:, :, 2] = xCenter + self.width / 2.0
        newPosition[:, :, 3] = yCenter + self.height / 2.0
        
        return newPosition
        
    
    """
    Plot (inside) the position in a frame. 

    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: [integer, integer]
    @param position: The objects's centroid coordinates (x, y)
    @type  outline:  string
    @param outline:  The name of the position color.
    """ 
    def plot(self, frame, position, outline):
        draw = ImageDraw.Draw(frame)
        w, h = frame.size
        x = position[0]
        y = position[1]
        draw.line([x, 0, x, h], fill=outline)
        draw.line([0, y, w, y], fill=outline)