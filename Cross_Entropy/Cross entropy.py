#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
A = list()
B = list()
C = list()
sum=0
for i in range(3):
    P=input("enter probability")
    Y=input("enter present")
    A.append(P) 
    B.append(Y) 
    
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    Z = -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
    return Z
for j in range(3):
    print(cross_entropy(B[j],A[j]))
    sum= sum+cross_entropy(B[j],A[j])
    
Entropy =sum
print(Entropy)  
                 

