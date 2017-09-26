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
n_iters=1000
np.random.seed(1)
X = np.array([  [1,2,3],
                [4,5,2],
                [6,3,2],
                [1,1,1]
            ])
Y = np.array([[6],
			 [40],
			 [36],
			 [1]])
err=[]
wt0 = 2*np.random.random((3,4)) - 1
wt1 = 2*np.random.random((4,1)) - 1

for i in xrange(n_iters):
    l0 = X
    l1 = sigmoid(np.dot(l0,wt0))
    l2 = sigmoid(np.dot(l1,wt1)) #Uncomment for Classification
    #l2 = np.dot(l1,wt1) #Uncomment for Regression
    
    l2_err = Y - l2
    
    l2_grad = l2_err * sigderiv(np.dot(l1,wt1)) #Classification grad
    
    #l2_grad = l2_err * 0.01 #Regression
    
    l1_err = np.dot(l2_err,wt1.T)
    l1_grad = l1_err * sigderiv(np.dot(l0,wt0))
    
    wt0 = wt0 + np.dot(l0.T,l1_grad)
    wt1 = wt1 + np.dot(l1.T,l2_grad)
    if i%100==0 and i<500000:
        print np.mean(np.abs(l2_err))
        print l2_err
        print l2_grad
        print l1_err
        print l1_grad
        print " "
        err.append(np.mean(np.abs(l2_err)))
        

print "Op after training"
np.set_printoptions(suppress=True)
print l2
err = np.asarray(err)
mplt.plot(err)
mplt.show()
print err

