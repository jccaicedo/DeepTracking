# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:31:07 2016

@author: MindLAB
"""

import numpy as NP
import numpy.linalg as NLA
from keras.models import Sequential
from keras.layers import InputLayer
from tracking.model.core.Processor import Processor
from tracking.model.keras.SpatialTransformer import SpatialTransformer
from tracking.util.model.CentroidHWPM import CentroidHWPM


class Cropper(Processor):
    
    def __init__(self, frameDims, context, positionModel, distorsion):
        self.transformer = self.createTransformer(frameDims)
        self.context = context
        self.positionModel = positionModel
        self.chwPM = CentroidHWPM()
        self.distorsion = distorsion
    
    
    # position must be between [-1,1]
    # cropPosition.shape = (batchSize, seqLength, targetDim)
    # objPosition.shape = (batchSize, seqLength, targetDim)
    # frame.shape = (batchSize, seqLength, channels, height, width)
    def crop(self, frame, cropPosition, objPosition):
        batchSize, seqLength, channels, height, width = frame.shape
        
        # Generating the transformations
        theta, thetaInv = self.generateTheta(cropPosition)
        
        # Generating the frames crops
        frame = frame.reshape((batchSize * seqLength, channels, height, width))
        frame = self.transformer.predict_on_batch([frame, theta])
        frame = frame.reshape((batchSize, seqLength, channels, height, width))
        
        # Generating the positions
        objPosition = self.positionModel.transform(thetaInv, objPosition)
        
        return frame, objPosition, theta
        
        
    def generateTheta(self, position):
        batchSize, seqLength, targetDim = position.shape
        samples = batchSize * seqLength
        theta = NP.zeros((samples, 3, 3), dtype='float32')
        
        targetDim = self.chwPM.getTargetDim()
        chw = self.chwPM.fromTwoCorners(position)
        chw = chw.reshape((samples, targetDim))
        dx = NP.random.uniform(-self.distorsion, self.distorsion, size=(samples))
        dy = NP.random.uniform(-self.distorsion, self.distorsion, size=(samples))
        cX = chw[:, 0] + dx
        cY = chw[:, 1] + dy
        h = chw[:, 2] * self.context + 0.01
        w = chw[:, 3] * self.context + 0.01
        maxW = 1.0-NP.abs(cX)
        maxH = 1.0-NP.abs(cY)
        
        # Calculating the parameters of the transformation
        tx = cX
        ty = cY
        sx = NP.minimum(w / 2.0, maxW) # Scale x
        sy = NP.minimum(h / 2.0, maxH) # Scale y
        
        # Setting transformation
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        theta[:, 2, 2] = 1.0
        
        return theta[:, :2, :], NLA.inv(theta)[:, :2, :]
        
        
    def createTransformer(self, frameDims):
        thetaModel = Sequential()
        thetaModel.add(InputLayer(input_shape=(2, 3)))
        
        frameModel = Sequential()
        frameModel.add(InputLayer(input_shape=frameDims))
        
        transformer = Sequential()
        transformer.add(SpatialTransformer([frameModel, thetaModel], 1.))
        transformer.compile(optimizer="rmsprop", loss='mse')
        
        return transformer