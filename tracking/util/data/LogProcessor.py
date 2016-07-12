# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:18:20 2016

@author: MindLab
"""

def getValues(logPath):
    epochMeasures = []
    batchMeasures = []
    batchLosses = []
    
    with open(logPath, mode="r") as log:
        for line in log:
            if "Validation Epoch:" in line:
                value = float(line.split("=")[-1])
                epochMeasures.append(value)
                
            if "Validation Batch:" in line:
                value = float(line.split("=")[-1])
                batchMeasures.append(value)
        
            if "Batch Loss:" in line:
                loss = line.split(":")[-1].split(",")[-1]
                loss = float(loss.split("=")[-1])
                batchLosses.append(loss)
                
    return batchLosses, batchMeasures, epochMeasures