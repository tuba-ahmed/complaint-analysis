#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# In[3]:


# Loading the dataset
df = pd.read_csv("complaints.csv")


# ### Initial Investigation for the dataset that we have selected. 
# 
# First, we are trying to figure out the structure of the data. Seeing how the dataset looks like and what attributes it has. 
# Just by looking at the attributes and our end goal, we can determine which of these attributes we can actually use (be beneficial to us)

# In[4]:


df.shape


# In[5]:


df.info()
df.head()


# Here, we try to determine the unique values and null values to determine if the dataset has enough representative data for our goal

# In[6]:


# Printing the counts of each column (unique and null vals.)
for col in df.select_dtypes(['object']).columns:
    print('Col:', col,'; Num. unique :',df[col].nunique(),'; Num. NaN :',df[col].isna().sum())


# In[29]:


df["Product"].value_counts()


# Similar to above, we try to see if the type of complaints are not saturated around one few types of products. This graph shows us that the complaints, although mostly around Credit reporting, credit repair services and other personal consumer reports, are decent enough to be representative of complaints.

# In[30]:


# Plotting the countplot for the product determining the categeory and their counts
sb.countplot(data=df, y = "Product")
plt.show()


# In[59]:


# Plotting the countplot for Submitted via
plt.figure(figsize=(15,20))
ax = sb.countplot(data=df, y = "State")
#plt.figure(figsize=(15,20))
plt.show()


# In[58]:


# We can seee that there is a high percentage of null values
df.isnull().mean(0).plot.bar()


# Here, we try to reduce the number of attributes, dropping off the ones that are irrelevant to our work. Below is the updated data.

# In[31]:


# Keeping the relevant attributes from the dataset
df_less = df[['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative', 'Company public response', 'Company', 'State','Submitted via', 'Company response to consumer', 'Consumer disputed?']]
df_less.shape
df_less.head()


# Here, we will calculate the percentage of data that actually has text in the 'Consumer complaint narrative'. This will tell us how much of the data is actually meaningful to us. 
# 
# We see that around 35.9 percentage of data is meaningful to us out of the entire dataset. Considering that the data has 3041367 lines, 35.9% is approx. 1091850.753 which is still significantly meaningful to our work. Since we are performing text analysis, we should be able to work on this dataset.

# In[32]:


# Since we are doing text analysis using mostly the consumer complaint narrative, we will drop the rows where val. for narrative is NaN
df_less['Consumer complaint narrative'].notnull().sum() / len(df) * 100


# Here, we get rid of the data that has none (NaN) in the 'Consumer complaint narrative'. This will make our process more efficeient as we don't have to go through the redundant data. Number of lines was reduced from 3041367 to 1091487. 

# In[33]:


df_less_1 = df_less[pd.notnull(df_less['Consumer complaint narrative'])]
df_less_1.shape


# In[34]:


df_less_1.head()


# In[35]:


df_less_1['Consumer complaint narrative'].notnull().sum() / len(df) * 100


# In[47]:


df_less_1.describe()


# In[51]:


# Determining the missing values
df_less_1.isnull().sum(0)


# #### Patterns
# 
# One of the pattern that I see for this is a lot of attributes for the data have missing values. As seen below from the percentage bar plot. 

# In[57]:


# Determining the missing values (percentage)
df_less_1.isnull().mean(0).plot.bar()


# #### Challenges
# 
# One of the challenges that I see that can occue in the future is that since certain attributes have missing values, certain Machine Learning algorithms do not accept missing values. We can see that certain attributes like 'Consumer disputed?' has a huge percentage of missing vlaues. So even if we try to fill in the missing values with something, it would result in inaccurate data. 

# In[ ]:




