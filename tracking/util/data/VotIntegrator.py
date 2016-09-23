# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:30:52 2016

@author: MindLAB
"""

import numpy as NP
import sys
import scipy.misc as SCPM
import tracking.util.data.Preprocess as Preprocess
import tracking.util.data.VotTool as VotTool
import vot
from tracking.model.keras.Tracker import Tracker
from tracking.util.data.GeneralProcessor import GeneralProcessor
from tracking.util.model.TwoCornersPM import TwoCornersPM


def loadTracker(path="/home/fhdiaze/Models/Tracker22.pkl"):
    tracker = Tracker()
    tracker.load(path)
    
    return tracker


def track(tracker, processor, frame, position, size):

    x, y, w, h = position
    x1, y1 = x + w, y + h

    position = NP.array([x, y, x1, y1])
    position = NP.expand_dims(position, axis=0)
    position = NP.expand_dims(position, axis=1)
    
    originalSize = frame.shape[:2][::-1] # imageSize must be (width, height)
    frame = SCPM.imresize(frame, size)
    frame = NP.expand_dims(frame, axis=0)
    frame = NP.expand_dims(frame, axis=1)
    
    position = Preprocess.scalePosition(position, originalSize)
    position = Preprocess.rescalePosition(position, size)

    frame, position = processor.preprocess(frame, position)
    position = tracker.forward([frame], position[:, 0, :])
    
    x, y, x1, y1 = position[0, 0, :]
    
    return vot.Rectangle(x, y, x1 - x, y1 - y)
    
    
def main():
    frameDims = (3, 224, 224) # (channels, width, height)
    
    positionRepresentation = TwoCornersPM()
    processor = GeneralProcessor(frameDims[1:3], positionRepresentation)
    
    handle = vot.VOT("rectangle")
    position = handle.region() # region = (topLeftX, topLeftY, width, height)
    
    framePath = handle.frame()
    
    if not framePath:
        sys.exit(0)
    
    frame = VotTool.loadFrame(framePath)
    tracker = loadTracker()
    
    while True:
        framePath = handle.frame()
        
        if not framePath:
            break
    
        frame = VotTool.loadFrame(framePath)
        position = track(tracker, processor, frame, position, frameDims[1:3])
        
        
        handle.report(position)


if __name__ == "__main__":
    main()