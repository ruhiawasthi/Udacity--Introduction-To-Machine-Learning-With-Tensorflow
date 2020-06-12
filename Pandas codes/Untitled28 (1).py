#!/usr/bin/env python
# coding: utf-8

# In[4]:


import  pandas as pd
groceries = pd.Series(data=[30,6,'yes','no'],index=['egges','apple','milk','bread'])
print(groceries)
print(groceries['egges'])
print(groceries[['milk','bread']])
print(groceries.loc[['egges','apple']])
print(groceries[[0,1]])
print(groceries[[-1]])
print(groceries[0])
print(groceries.iloc[[2,3]])

