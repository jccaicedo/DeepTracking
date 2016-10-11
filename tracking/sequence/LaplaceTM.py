# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:29:48 2016

@author: MindLAB
"""

import numpy as NP
from tracking.util.model.CentroidHWPM import CentroidHWPM

class LaplaceTM:
    
    def __init__(self, muX, lambdaX, muS, lambdaS):
        self.samplerP = lambda size : NP.random.laplace(loc=muX, scale=lambdaX, size=size)
        self.samplerS = lambda size : NP.random.laplace(loc=muS, scale=lambdaS, size=size)
        self.centroidHWPM = CentroidHWPM()
       
       
    def generateTrajectory(self, initP, length):
        position = NP.zeros((length, self.centroidHWPM.getTargetDim()))
        initP = self.centroidHWPM.fromTwoCorners(initP)
        theta = NP.zeros((length, 3, 3), dtype='float32')
        
        # Initialize the trajectory
        position[0, ...] = initP
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0
        theta[:, 2, 2] = 1.0
        
        for t in range(1, length):
            deltaP = position[t-1, [2, 3]] * self.samplerP((1, 2))
            deltaS = NP.clip(self.samplerS((1, 2)), 0.6, 1.4)
            
            thetaC = NP.zeros((3, 3))
            thetaC[[0, 1, 2], [0, 1, 2]] = 1.0
            thetaC[[0, 1], [2, 2]] = position[t-1, [0, 1]]
            
            thetaZ = NP.zeros((3, 3))
            thetaZ[[0, 1], [0, 1]] = 1.0 / deltaS[:, ::-1]
            thetaZ[2, 2] = 1.0
            
            thetaU = NP.zeros((3, 3))
            thetaU[[0, 1, 2], [0, 1, 2]] = 1.0
            thetaU[[0, 1], [2, 2]] = -position[t-1, [0, 1]]
            
            thetaT = NP.zeros((3, 3))
            thetaT[[0, 1, 2], [0, 1, 2]] = 1.0
            thetaT[[0, 1], [2, 2]] = -deltaP
            
            # Setting transformation
            theta[t, ...] = NP.dot(NP.dot(NP.dot(thetaC, thetaZ), thetaU), thetaT)
            position[t, [0, 1]] = position[t-1, [0, 1]] + deltaP
            position[t, [2, 3]] = position[t-1, [2, 3]] * deltaS
 
        position = self.centroidHWPM.toTwoCorners(position)
        return theta, position