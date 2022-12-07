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
import seaborn as sb
import missingno as msno
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[2]:


#Read new processed complaint data set.
ndf = pd.read_csv('processed_complaints.csv')


# In[3]:


pd.set_option('display.max_colwidth', -1)


# # Initial data analysis.

# In[4]:


#Examining dataframe.
ndf


# In[5]:


#Eliminate 'Unnamed:0' column.
ndf.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[6]:


#Check how many null values are in the Complaint column.
print('Product Nulls: ', ndf['Product'].isnull().sum())
print('Complaint Nulls: ', ndf['Complaint'].isnull().sum())


# In[7]:


#Visualize null and non-null values in product and complaint features.
msno.matrix(ndf)


# In[8]:


#Sample narrative after preprocessing.
ndf.Complaint[0]


# In[9]:


#Eliminate all null values.
ndf = ndf.dropna()


# In[10]:


ndf.head()


# In[11]:


ndf.shape


# In[12]:


# Plotting the countplot for 'product' column
sb.countplot(data=ndf, y='Product')


# # Encoding products.

# In[13]:


#View list of products in Product feature.
ndf['Product'].unique()


# In[14]:


#Encode products in numeric values.
ndf['Product'].replace({'Debt Collection' : 0, 
                        'Credit Reporting and Services' : 1,
                        'Banking Services' : 2,
                        'Mortgages' : 3,
                        'Credit/Prepaid Cards' : 4,
                        'Loans' : 5,
                        'Crypto Currency' : 6}, inplace=True)


# # Generate training and test set.

# In[15]:


#Create train and test sets.
x = ndf['Complaint']
y = ndf['Product']


# In[16]:


#Train and test sets are split 80/20.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[17]:


print("Training set size:" , x_train.shape)
print("Test set size:" , x_test.shape)


# # Using TF-IDF Vectorizer to obtain relevance of each word in complaint narratives.

# In[18]:


tfidf = TfidfVectorizer(max_features=900000, ngram_range=(1,2))
tfidf_x_train = tfidf.fit_transform(x_train)
tfidf_x_test = tfidf.transform(x_test)


# # Model construction using Multinominal Naive Bayes classifier.

# In[19]:


nb = MultinomialNB()
nb.fit(tfidf_x_train, y_train)


# In[20]:


#Predictions for training and testing set.
pred_y_train = nb.predict(tfidf_x_train)
pred_y_test = nb.predict(tfidf_x_test)


# In[21]:


train_score = accuracy_score(y_train, pred_y_train)
test_score = accuracy_score(y_test, pred_y_test)

print('Prediction Accuracy (Train): ', 100*(round(train_score, 2)), '%')
print('Prediction Accuracy (Test): ', 100*(round(test_score, 2)), '%')


# # Classification details. 

# In[22]:


def getMetrics(org_data, pred_data):
    precision = precision_score(org_data, pred_data, average='weighted')
    recall = recall_score(org_data, pred_data, average='weighted')
    f1 = f1_score(org_data, pred_data, average='weighted')
    accuracy = accuracy_score(org_data, pred_data)
    
    print("Metrics:\n")
    print("   Precision: ", round(precision, 4),
         "\n   Recall: ", round(recall, 4),
         "\n   F1-Score: ", round(f1, 4),
         "\n   Accuracy: ", round(accuracy, 4))


# In[23]:


#Generates confusion matrix.
def genCMatrix(org_data, pred_data):
    cmat = confusion_matrix(org_data, pred_data)

    disp_fig = metrics.ConfusionMatrixDisplay(confusion_matrix = cmat)
    disp_fig.plot()
    
    print('\n\nConfusion Matrix:')
    plt.show()


# In[24]:


train_clsf_report = classification_report(y_train, pred_y_train)
test_clsf_report = classification_report(y_test, pred_y_test)


# ### Training set evaluation.

# In[25]:


getMetrics(y_train, pred_y_train)
print('\n\n\nClassification Report:\n\n', train_clsf_report)
genCMatrix(y_train, pred_y_train)


# ### Test set evaluation.

# In[26]:


getMetrics(y_test, pred_y_test)
print('\n\nClassification Report:\n\n', test_clsf_report)
genCMatrix(y_test, pred_y_test)


# In[ ]:




