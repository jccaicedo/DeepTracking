# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:02:40 2016

@author: MindLab
"""

import theano.tensor as THT


"""
Creates a theano tensor with specif dimensions

@type    name:         String
@param   name:         The name for the tensor
@type    dtype:        dtype
@param   dtype:        The type of the tensor
@type    dim:          dtype
@param   dim:          The type of the tensor
@rtype:  theano.tensor
@return: The theano tensor
""" 
def getTensor(name, dtype, dim):
    
    return THT.TensorType(dtype, [False] * dim, name=name)()