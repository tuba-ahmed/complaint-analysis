#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("complaints.csv")


# In[5]:


chunksize = 10 ** 6
with pd.read_csv("complaints.csv", chunksize=chunksize) as reader:
    for chunk in reader:
        process(chunk)


# In[6]:


df.head()


# In[7]:


df.columns


# In[8]:


df['Product'].value_counts()


# In[9]:


df['Sub-product'].value_counts()


# In[10]:


df['Issue'].value_counts()


# In[13]:


df['Sub-issue'].value_counts()


# In[14]:


df['Consumer complaint narrative'].value_counts()


# In[11]:


df.shape


# In[15]:


df['Company public response'].value_counts()


# In[28]:


df['Company'].value_counts()


# In[29]:


df['Company response to consumer'].value_counts()


# In[30]:


df['Timely response?'].value_counts()


# In[31]:


df_less = df[['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative', 'Company public response', 'Company', 'State','Submitted via', 'Company response to consumer', 'Consumer disputed?']]
df_less.shape
df_less.head()


# In[35]:


df_less.groupby("Product").agg({'Sub-product':'nunique'}).reset_index()


# In[37]:


df_less.groupby("Product").agg({'Consumer complaint narrative':'nunique'}).reset_index()


# In[38]:


df_less.groupby("Product").agg({'Consumer complaint narrative':'size'}).reset_index()


# In[39]:


df_less.groupby("Issue").agg({'Sub-issue':'nunique'}).reset_index()


# In[63]:


df_na = df_less[df_less['Consumer complaint narrative'].isna()]
df_nona = df_less[df_less['Consumer complaint narrative'].isna() == False]


# In[43]:


df_less.groupby("Product").agg({'Issue':'unique'}).reset_index()


# In[47]:


df_less[df_less['Issue'] == "Other transaction issues"]['Product'].unique()


# In[51]:


df_less[df_less['Product'] == "Credit card or prepaid card"]['Sub-product'].value_counts()


# In[52]:


df_less[df_less['Product'] == "Prepaid card"]['Sub-product'].value_counts()


# In[56]:


df_less[df_less['Product'] == "Credit card"]['Sub-product'].value_counts()


# In[62]:


df_na['Product'].value_counts()


# In[64]:


df_nona['Product'].value_counts()


# In[60]:


df.shape


# In[61]:


df.sample(10)


# In[ ]:




