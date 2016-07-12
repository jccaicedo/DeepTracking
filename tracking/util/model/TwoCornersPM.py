# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:37:32 2016

@author: MindLab
"""

from PIL import ImageDraw
from tracking.model.core.PositionModel import PositionModel

class TwoCornersPM(PositionModel):
    
    def __init__(self):
        super(TwoCornersPM, self).__init__(4)
    
    
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def fromTwoCorners(self, position):
        
        return position
    
    
    # position.shape = (batchSize, seqLength, targetDim(xC, yC, height, width))
    def toTwoCorners(self, position):
        
        return position
    
    """
    Plot (in side) the position in a frame. 

    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: [integer, integer, integer, integer]
    @param position: The objects's corners coordinates [x1, y1, x2, y2]
    @type  outline:  string
    @param outline:  The name of the position color.
    """ 
    def plot(self, frame, position, outline):
        draw = ImageDraw.Draw(frame)
        
        draw.rectangle(position, outline=outline)