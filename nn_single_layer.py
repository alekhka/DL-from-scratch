#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 11:35:56 2017

@author: alekh
"""

import numpy as np
import matplotlib.pyplot as mplt #Used to plot error over iterations remove if not needed

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigderiv(x):
    y = 1/(1+np.exp(-x))
    return y*(1-y)

n_iters=10000
np.random.seed(1)
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]
        ])
Y = np.array([[0,0,1,1]]).T
err=[]
wt3 = 2*np.random.random((3,1)) - 1

for i in xrange(n_iters):
    l0 = X

    #l1 = np.dot(l0,wt0) #Uncomment this for Regression
    l1 = sigmoid(np.dot(l0,wt3)) #Uncomment this for Classification
    l1_err = Y - l1
    
    #l1_grad = l1_err * 0.001 #Gradient in case of regrassion
    l1_grad = l1_err * sigderiv(np.dot(l0,wt3)) #Gradient in case of Classification
    
    wt3 = wt3 + np.dot(l0.T,l1_grad)
    
    if i%1==0 and i<5000:
        err.append(np.sum(np.abs(l1_err)/4))

print "Op after training"
np.set_printoptions(suppress=True)
print l1
err = np.asarray(err)
mplt.plot(err)
mplt.show()
print wt3
