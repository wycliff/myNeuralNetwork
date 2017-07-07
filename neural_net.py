    # -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 21:49:08 2017

@author: Wycliffe
"""
    
import numpy as np
from math import *

#Sigmoid function that maps a value to any  value between 0 and 1
#this function is run at every neuron of our   network wen data hits it
# Useful for creating probabilities out of numbers 
def nonlin(x,deriv= False):
    if(deriv == True):
        return x*(1-x)
        
    return 1/(1+np.exp(-x))
 
#input data : matrix
#each row : training example 
#each column : neuron/ input node 
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
              
# Output data
y =  np.array([[0],
               [1],
               [1],
               [0]])

#seed(same starting point) random numbers to make them deterministic               
np.random.seed(1)
    
#synapse matrices (weights/ connections)
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

#The training code
for j in xrange(60000):
    
    #The layers, prediction happens here 
    
    l0 = x
    #matrix multiplication by weight then passed through the sigmoid function 
    l1 = nonlin(np.dot(l0,syn0))
    # l2 gives the final value
    l2 = nonlin(np.dot(l1,syn1))
    
    #the error in prediction
    l2_error = y-l2
    
    #print out average error rate at set interval 1000 step, to make sure it ges down every time
    if(j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))
    
    
     # multiply error rate by result of sigmoid funct
     # gives a delta:used to reduce error rate of our predictions when synapses are updated every iteration.   
     # nonlin() here : give der of output funct from layer2
    l2_delta = l2_error * nonlin(l2,deriv = True)
    
    
    #BACK PRPAGATION:to see how much did error in layer 1 contribute to that in layer 2
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv = True)
    
    #Update synapse weights, to continually reduce the error rate more and more. (GRADIENT DESCENT)
    #Easy: multiply each layer by a delta
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("The output after Training")
print(l2)

        
          
        
        
        
        
         
            
        
        
    
    
        
        
               
        