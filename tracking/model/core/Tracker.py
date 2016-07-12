# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:12:34 2016

@author: MindLab
"""

import logging
import cPickle as pickle

class Tracker(object):
    
    """
    Make a fit process using a set of frames and positions.

    @type    frame: numpy.ndarray
    @param   frame: The frames (batchSize, seqLength, channels, height, width)
    @type    position: numpy.ndarray
    @param   position: The positions over the frames (batchSize, seqLength, targetDim)
    @type    lnr: float
    @param   lnr: The learning rate
    @rtype:  (float, numpy.ndarray)
    @return: The cost in training
    """
    def fit(self, frame, position):
        pass
    
    
    """
    Make a forward process using a set of frames and positions.

    @type    frame: numpy.ndarray
    @param   frame: The frames (batchSize, seqLength, channels, height, width)
    @type    initPosition: numpy.ndarray
    @param   initPosition: The positions in the first frames (batchSize, targetDim)
    @rtype:  (float, numpy.ndarray)
    @return: The cost in training
    """
    def forward(self, frame, initPosition):
        pass
    
    
    """
    Boolean (default False). If True, the last state for each sample at index i
    in a batch will be used as initial state for the sample of index i in the 
    following batch

    @type    stateful: boolean
    @param   stateful: stateful value
    """
    def setStateful(self, stateful):
        pass
    
    
    def reset(self):
        pass
    
    
    def save(self, path):
        logging.info("Saving tracker to %s", path)
        modelFile = open(path, 'wb')
        pickle.dump(self.__dict__, modelFile, protocol=pickle.HIGHEST_PROTOCOL)
        modelFile.close()
            
    
    def load(self, path):
        logging.info("Loading tracker from %s", path)
        modelFile = open(path, 'rb')
        tmpDict = pickle.load(modelFile)
        modelFile.close()
        self.__dict__.update(tmpDict)