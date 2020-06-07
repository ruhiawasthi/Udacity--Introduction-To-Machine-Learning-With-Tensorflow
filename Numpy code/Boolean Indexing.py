#!/usr/bin/env python
# coding: utf-8

# In[7]:


# We create a 5 x 5 ndarray that contains integers from 0 to 24
import numpy as np
X = np.arange(25).reshape(5, 5)

# We print X
print('Original X = \n', X)

# We use Boolean indexing to select elements in X:
print('The elements in X that are greater than 10:', X[X > 10])
print('The elements in X that less than or equal to 7:', X[X <= 7])
print('The elements in X that are between 10 and 17:', X[(X > 10) & (X < 17)])

# We use Boolean indexing to assign the elements that are between 10 and 17 the value of -1
X[(X > 10) & (X < 17)] = -1

# We print X
print('X = \n', X)
 
# We create a rank 1 ndarray
x = np.array([1,2,3,4,5])

# We create a rank 1 ndarray
y = np.array([6,7,2,8,4])

# We print x
print('x = ', x)

# We print y
print('y = ', y)

# We use set operations to compare x and y:
print('The elements that are both in x and y:', np.intersect1d(x,y))
print('The elements that are in x that are not in y:', np.setdiff1d(x,y))
print('All the elements of x and y:',np.union1d(x,y))

