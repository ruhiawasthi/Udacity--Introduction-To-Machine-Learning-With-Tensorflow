#!/usr/bin/env python
# coding: utf-8

# In[3]:


# We create a 5 x 5 ndarray that contains integers from 0 to 24
import numpy as np
X = np.arange(25).reshape(5, 5)

# We print X
print()
print('Original X = \n', X)
print()

# We use Boolean indexing to select elements in X:
print('The elements in X that are greater than 10:', X[X > 10])
print('The elements in X that less than or equal to 7:', X[X <= 7])
print('The elements in X that are between 10 and 17:', X[(X > 10) & (X < 17)])
# We use Boolean indexing to assign the elements that are between 10 and 17 the value of -1
z= X[(X > 10) & (X < 17)] = -1
print(X)


