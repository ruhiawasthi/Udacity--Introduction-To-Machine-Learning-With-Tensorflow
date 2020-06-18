#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data ={'Bob': pd.Series ([245,25,55]),'Alice': pd.Series ([40,110,500,45])}
df = pd.DataFrame(data)
print(df)

