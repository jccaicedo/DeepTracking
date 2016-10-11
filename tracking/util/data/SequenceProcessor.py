# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:09:49 2016

@author: MindLAB
"""

import numpy as NP

class SequenceProcessor(object):
    
    def before(self, input, output):
        if len(output.shape) < 3:
            output = NP.expand_dims(output, axis=1)
            
        return input  + [output, ]

    
    def after(self, output):
        
        return output