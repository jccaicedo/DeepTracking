# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:21:03 2016

@author: MindLab
"""
import numpy as NP

class RandomTrajectoryModel:
    
    # sceneSize = (w, h)
    def __init__(self, sceneSize, stepLength, velScale, accScale, scaleRange):
        self.sceneSize = sceneSize
        self.stepLength = stepLength
        self.velScale = velScale
        self.accScale = accScale
        self.scaleRange = scaleRange
        
        
    def getTrajectory(self, length):
        # Initial position uniform random inside the box.
        positions = NP.random.rand(2, 1)
        scales = NP.array([1.0])

        # Choose a random velocity.
        theta = NP.random.rand() * 2 * NP.pi
        initialVelocity = NP.random.normal(0, self.velScale, size=(2))
        velocity = initialVelocity * NP.array([NP.sin(theta), NP.cos(theta)])
        position =  positions[:, 0]
        
        for t in range(1, length):
            scale = NP.exp(self.scaleRange * (NP.random.random_sample()-0.5))
            position = position + self.stepLength * velocity
            
            # Bounce off edges.
            if position[0] <= 0.0:
                position[0] = 0.0
                velocity[0] = -velocity[0]
            if position[0] >= 1.0:
                position[0] = 1.0
                velocity[0] = -velocity[0]
            if position[1] <= 0.0:
                position[1] = 0.0
                velocity[1] = -velocity[1]
            if position[1] >= 1.0:
                position[1] = 1.0
                velocity[1] = -velocity[1]
            
            # Set the new position.
            positions = NP.hstack([positions, NP.matrix([position]).T])
            scales = NP.hstack([scales, [scale]])
            
            # Update the velocity
            velocity += NP.zeros(shape=(2)) if self.accScale == 0 else NP.random.normal(0, self.accScale, size=(2))
            
        # Scale to the size of the canvas.
        positions = NP.multiply(NP.matrix([self.sceneSize]).T, positions).astype(NP.int32)
        
        return positions, scales