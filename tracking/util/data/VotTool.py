# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:59:29 2016

@author: MindLab
"""

import os
import random
import numpy as NP
import cPickle as pickle
import scipy.misc as SCPM
import matplotlib.pyplot as PLT
import tracking.util.data.Preprocess as Preprocess
from PIL import Image


def loadFrame(path):
    frame = NP.array(Image.open(path))
    
    return frame


def loadPosition(path):
    # Load bounding boxes information
    position = []
    
    with open(path) as f:
        [position.append(list(map(float, line.split(",")))) for line in f]
    
    position = NP.array(position)[:, [2, 3, 6, 7]]
    
    return position
    

def loadSequence(path, extension, boxesFile, size):
    # Load frames
    framesPath = [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(extension)]
    framesPath = sorted(framesPath)
    frame = []
    
    
    for i, framePath in enumerate(framesPath):
        tmpFrame = loadFrame(framePath)
        originalSize = tmpFrame.shape[:2][::-1] # imageSize must be (width, height)
        frame.append(SCPM.imresize(tmpFrame, size))
    
    frame = NP.array(frame, dtype=NP.float)
    
    # Load bounding boxes information
    boxesPath = os.path.join(path, boxesFile)
    position = loadPosition(boxesPath)
    position = Preprocess.scalePosition(position, originalSize)
    position = Preprocess.rescalePosition(position, size)
    
    return frame, position
    
    
def loadSequences(path, names, extension, boxesFile, size):
    for name in names:
        seqPath = os.path.join(path, name)
        yield (name, ) + loadSequence(seqPath, extension, boxesFile, size)
    
    
def createDataset(path, seqs, extension, boxesFile, size, samples, seqLength, outPath):
    frames = NP.empty((samples, seqLength) + size + (3, ) )
    positions = NP.empty((samples, seqLength, 4))
    
    for i in range(samples):
        seqPath = os.path.join(path, random.choice(seqs))
        frame, position = loadSequence(seqPath, extension, boxesFile, size)
        start = NP.random.randint(0, frame.shape[0] - seqLength)
        frames[i, ...] = frame[start:start+seqLength, ...]
        positions[i, ...] = position[start:start+seqLength, ...]
        
    dataSetFile = open(outPath, "wb")
    pickle.dump((frames, positions), dataSetFile, protocol=pickle.HIGHEST_PROTOCOL)
    dataSetFile.close()
    

def loadDataset(path):
    dataSetFile = open(path, "rb")
    frame, position = pickle.load(dataSetFile)
    dataSetFile.close()
    
    return frame, position
    

def plotResults(measures, measureName):
    cols = 2
    rows = len(measures) / cols + (len(measures) % cols > 0)
    f, axs = PLT.subplots(rows, cols, figsize=(20, 28))
    axs = axs.ravel()
    i = 0
    for name, values in measures.iteritems():
        ax = axs[i]
        ax.plot(values[0, ...])
        value = str(values[0, ...].mean())
        title = "{}, {} {} = {}".format(name, measureName, "Mean", value)
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel(measureName)
        print title
        i += 1