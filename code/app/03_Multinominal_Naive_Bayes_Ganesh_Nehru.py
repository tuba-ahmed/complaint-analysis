#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


#Read new processed complaint data set.
ndf = pd.read_csv('processed_complaints.csv')


# In[3]:


#Examining dataframe.
ndf


# In[4]:


#Eliminate 'Unnamed:0' column.
ndf.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[5]:


#Check how many null values are in the Complaint column.
print(ndf['Product'].isnull().sum())
print(ndf['Complaint'].isnull().sum())


# In[6]:


#Eliminate all null values.
ndf = ndf.dropna()


# In[8]:


ndf = ndf.sample(n=100000, random_state=42)


# In[9]:


ndf


# In[10]:


ndf.shape


# # Encoding Product feature.

# In[11]:


#View list of products in Product feature.
ndf['Product'].unique()


# In[12]:


#Encode products in numeric values.
ndf['Product'].replace({'Debt Collection' : 0, 
                        'Credit Reporting and Services' : 1,
                        'Banking Services' : 2,
                        'Mortgages' : 3,
                        'Credit/Prepaid Cards' : 4,
                        'Loans' : 5,
                        'Crypto Currency' : 6}, inplace=True)


# In[13]:


ndf.isna().sum()


# # Generate training and test set.

# In[15]:


#Create train and test sets.
x = ndf['Complaint']
y = ndf['Product']


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


# In[17]:


print("Training set size:" , x_train.shape)
print("Test set size:" , x_test.shape)


# # Obtaining relevance of words from complaint narratives.

# In[18]:


tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000000)
tfidf_x_train = tfidf.fit_transform(x_train)
tfidf_x_test = tfidf.transform(x_test)


# # Modeling with Multinominal Naive Bayes classifier.

# In[19]:


nb = MultinomialNB()
nb.fit(tfidf_x_train, y_train)


# In[20]:


#Predictions for training and testing set.
pred_y_train = nb.predict(tfidf_x_train)
pred_y_test = nb.predict(tfidf_x_test)


# In[21]:


print('Training prediction accuracy: ', accuracy_score(y_train, pred_y_train))
print('Testing prediction accuracy: ', accuracy_score(y_test, pred_y_test))


# In[22]:


print('Classification Report for Naive Bayes (Training Data)\n', classification_report(y_train, pred_y_train))


# In[ ]:


cmat_train = confusion_matrix(y_train, pred_y_train)

cmat_train_disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cmat_train)
cmat_train_disp.plot()

print('Confusion Matrix for Naive Bayes (Train)')
plt.show()


# In[ ]:




