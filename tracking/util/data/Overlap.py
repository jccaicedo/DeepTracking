# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:46:25 2016

@author: MindLab
"""

import numpy as NP
from tracking.model.core.Measure import Measure

class Overlap(Measure):
    
    def __init__(self):
        super(Overlap, self).__init__("Overlap")
    
    # gtPosition.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def calculate(self, gtPosition, predPosition):
        left = NP.max([predPosition[..., 0], gtPosition[..., 0]], axis=0)
        top = NP.max([predPosition[..., 1], gtPosition[..., 1]], axis=0)
        right = NP.min([predPosition[..., 2], gtPosition[..., 2]], axis=0)
        bottom = NP.min([predPosition[..., 3], gtPosition[..., 3]], axis=0)
        intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
        label_area = NP.abs(gtPosition[..., 2] - gtPosition[..., 0]) * NP.abs(gtPosition[..., 3] - gtPosition[..., 1])
        predict_area = NP.abs(predPosition[..., 2] - predPosition[..., 0]) * NP.abs(predPosition[..., 3] - predPosition[..., 1])
        union = label_area + predict_area - intersect
        iou = intersect / union
                
        return iou