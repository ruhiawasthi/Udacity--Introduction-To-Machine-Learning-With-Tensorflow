#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
y=np.array([1,2,4,5,6])
print(y)
y=np.insert(y,2,[3])
print(y)
x=np.array([[1,2,3],[7,8,9]])
print(x)
v=np.insert(x,1,[4,5,6],axis=0)
print(v)
w=np.insert(x,1,4,axis=1)
print(w)

