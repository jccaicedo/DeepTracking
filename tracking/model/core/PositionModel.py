# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:35:26 2016

@author: MindLab
"""

class PositionModel(object):
    
    def __init__(self, targetDim):
        self.targetDim = targetDim
        
    
    def getTargetDim(self):
        
        return self.targetDim
    
    def fromTwoCorners(self, position):
        pass
    
    
    def toTwoCorners(self, position):
        pass
    
    """
    Plot (inside) the position in a frame. 

    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: object
    @param position: The objects's position representation
    @type  outline:  string
    @param outline:  The name of the position color.
    """ 
    def plot(self, frame, position, outline):
        pass