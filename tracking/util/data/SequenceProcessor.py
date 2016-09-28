# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:09:49 2016

@author: MindLAB
"""

import numpy as NP

class SequenceProcessor(object):
    
    def before(self, frame, position):
        if len(position.shape) < 3:
            position = NP.expand_dims(position, axis=1)
            
        return [frame[0], position]

    
    def after(self, position):
        
        return position