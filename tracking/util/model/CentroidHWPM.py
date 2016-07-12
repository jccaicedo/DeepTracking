# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:38:27 2016

@author: MindLab
"""

import numpy as NP
from PIL import ImageDraw
from tracking.model.core.PositionModel import PositionModel

class CentroidHWPM(PositionModel):
    
    def __init__(self):
        super(CentroidHWPM, self).__init__(4)
        
        
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def fromTwoCorners(self, position):
        x1 = position[:, :, 0]
        y1 = position[:, :, 1]
        x2 = position[:, :, 2]
        y2 = position[:, :, 3]
        
        w = NP.abs(x2 - x1)
        h = NP.abs(y2 - y1)
        xCenter = x1 + w / 2.0
        yCenter = y1 + h / 2.0
        
        newPosition = NP.copy(position)
        
        newPosition[:, :, 0] = xCenter
        newPosition[:, :, 1] = yCenter
        newPosition[:, :, 2] = h
        newPosition[:, :, 3] = w
        
        return newPosition
    
    
    # position.shape = (batchSize, seqLength, targetDim(xC, yC, height, width))
    def toTwoCorners(self, position):
        xCenter = position[:, :, 0]
        yCenter = position[:, :, 1]
        h = position[:, :, 2]
        w = position[:, :, 3]
        
        newPosition = NP.copy(position)
        
        newPosition[:, :, 0] = xCenter - w / 2.0
        newPosition[:, :, 1] = yCenter - h / 2.0
        newPosition[:, :, 2] = xCenter + w / 2.0
        newPosition[:, :, 3] = yCenter + h / 2.0
        
        return newPosition
        
        
    """
    Plot (inside) the position in a frame. 

    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: [integer, integer, integer, integer]
    @param position: The objects's centroid coordinates, width and height [x1, y1, w, h]
    @type  outline:  string
    @param outline:  The name of the position color.
    """ 
    def plot(self, frame, position, outline):
        draw = ImageDraw.Draw(frame)
        xCenter = position[0]
        yCenter = position[1]
        h = position[2]
        w = position[3]
        x1 = xCenter - w / 2.0 
        y1 = yCenter - h / 2.0
        x2 = xCenter + w / 2.0
        y2 = yCenter + h / 2.0
        
        draw.rectangle([x1, y1, x2, y2], outline=outline)