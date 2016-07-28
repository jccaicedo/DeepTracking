# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:14:44 2016

@author: MindLab
"""

import h5py
from keras.models import Sequential
from tracking.model.keras.Cnn import Cnn
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

class VggCnn(Cnn):
    
    def __init__(self, modelPath, layerKey):
        self.layerKey = layerKey
        self.buildModel(modelPath, layerKey)        
    
        
    def buildModel(self, modelPath, layerKey):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        
        model = self.loadWeights(model, modelPath)
        
        self.model = model
        
        
    def loadWeights(self, model, modelPath):
        f = h5py.File(modelPath)
        
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        
        f.close()
        
        return model
        
    def getOutputDim(self, inDims):
        outDims = self.model.output_shape
        
        return outDims
        
        
    def getModel(self):
        return self.model
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable