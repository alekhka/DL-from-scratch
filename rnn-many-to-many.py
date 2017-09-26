#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 00:39:42 2017

@author: alekh
"""

import copy
import numpy as np

np.random.seed(0)

def sigmoid(x):
    op= 1/(1+np.exp(-x))
    return op

def sigderiv(x):
    return x*(1-x)

int2bin = {}
binary_dim = 8

largest_no = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_no)],dtype=np.uint8).T,axis=1)
for i in range(largest_no):
    int2bin[i]=binary[i]
    
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

wt0 = 2*np.random.random((input_dim,hidden_dim)) -1
wt1 = 2*np.random.random((hidden_dim,output_dim)) -1
wth = 2*np.random.random((hidden_dim,hidden_dim)) -1

wt0_update = np.zeros_like(wt0)
wt1_update = np.zeros_like(wt1)
wth_update = np.zeros_like(wth)

for j in range(1):
    
    aint = np.random.randint(largest_no/2)
    a = int2bin[aint]
    bint = np.random.randint(largest_no/2)
    b = int2bin[bint]

    cint = aint+bint
    c = int2bin[cint]
    
    d = np.zeros_like(c)
    
    totErr = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.atleast_2d(np.zeros(hidden_dim)))
    
    for pos in range(binary_dim):
        
        X = np.array([[a[binary_dim-pos-1],b[binary_dim-pos-1]]])
        y = np.array([[c[binary_dim-pos-1]]]).T
        
        layer_1 = sigmoid(np.dot(X,wt0)+np.dot(layer_1_values[-1],wth))
        
        layer_2 = sigmoid(np.dot(layer_1,wt1))
        
        l2_err = y - layer_2
        layer_2_deltas.append((l2_err)*sigderiv(layer_2))
        totErr = totErr + np.abs(l2_err[0])
        
        d[binary_dim-pos-1]=np.round(layer_2[0][0])
        
        layer_1_values.append(copy.deepcopy(layer_1))
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for pos in range(binary_dim):
       
        X = np.array([[a[pos],b[pos]]])
        print X
        layer_1 = layer_1_values[-pos-1]
        prev_layer_l = layer_1_values[-pos-2]
        
        layer_2_delta = layer_2_deltas[-pos-1]
        
        layer_1_delta = (np.dot(future_layer_1_delta,wth.T)+np.dot(layer_2_delta,wt1.T))*sigderiv(layer_1)
        
        
        wt1_update += np.atleast_2d(np.dot(layer_1.T,layer_2_delta))
        wth_update += np.atleast_2d(np.dot(prev_layer_l.T,layer_1_delta))
        wt0_update += np.atleast_2d(np.dot(X.T,layer_1_delta))

        future_layer_1_delta=layer_1_delta
        
    wt0 = wt0 + wt0_update * alpha
    wt1 = wt1 + wt1_update * alpha
    wth = wth + wth_update * alpha
    
    wt1_update *= 0
    wth_update *= 0
    wt0_update *= 0
    
    if(j % 1000 == 0):
        print "Error:" + str(totErr)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(aint) + " + " + str(bint) + " = " + str(out)
        print "------------"
        
        
        
        
        
        
        
        
        
        
        
        