#!/usr/bin/env python
# coding: utf-8

# In[35]:


from collections import Counter
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import lightgbm as lgb
import os


# In[36]:


os.listdir()


# In[50]:


# sub1 = pd.read_csv('submission_069859.csv')
# sub2 = pd.read_csv('submission_069812.csv')
sub3 = pd.read_csv('submission_06989.csv')
sub4 = pd.read_csv('submission_06988.csv')
sub5 = pd.read_csv('submission_0699.csv')
sub6 = pd.read_csv('submission_final.csv')


# In[54]:


df_all = pd.concat([sub3,sub4,sub5,sub6])


# In[55]:


pred = df_all.groupby('userid',as_index=False).label.mean()


# In[56]:


pred.to_csv('submission_ensem.csv', index=False)


# In[ ]:




