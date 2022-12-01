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
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


#Reading the dataset. 
df = pd.read_csv('complaints.csv')


# In[3]:


#Data frame before performing data cleanup.
df


# ##### Since we are only focused on the products and complaint narratives, we will re-create the data frame using only the 'Product' and 'Consumer complaint narrative' attributes.

# In[4]:


df = df[['Product' , 'Consumer complaint narrative']]


# In[5]:


#Rename attributes of dataframe for simplicity.
df = df.rename(columns={'Product' : "Product",
                       'Consumer complaint narrative' : "Complaint"})


# In[6]:


#Data frame after renaming attributes.
df


# In[7]:


df.groupby('Product').count()


# ##### Many of the products of the same category can be merged together into a single product category.

# In[8]:


#Clean up data by renaming similar products into one category of products.
df['Product'].replace({'Bank account or service': 'Banking Services',
                       'Checking or savings account' : 'Banking Services',
                       'Consumer Loan' : 'Loans',
                       'Credit card' : 'Credit/Prepaid Cards',
                       'Credit card or prepaid card' : 'Credit/Prepaid Cards',
                       'Credit reporting' : 'Credit Reporting and Services',
                       'Credit reporting, credit repair services, or other personal consumer reports' : 'Credit Reporting and Services',
                       'Debt collection' : 'Debt Collection',
                       'Money transfer, virtual currency, or money service' : 'Banking Services',
                       'Money transfers' : 'Banking Services',
                       'Mortgage' : 'Mortgages',
                       'Other financial service' : 'Banking Services',
                       'Payday loan' : 'Loans',
                       'Payday loan, title loan, or personal loan' : 'Loans',
                       'Prepaid card' : 'Credit/Prepaid Cards',
                       'Student loan' : 'Loans',
                       'Vehicle loan or lease' : 'Loans',
                       'Virtual currency' : 'Crypto Currency'}, inplace=True)


# #Refined products and their counts.
# df.groupby('Product').count()

# In[9]:


#Data frame with refined product categories.
df


# ### Cleanup data frame of null values.

# In[12]:


#Check number of null values.
df[pd.isnull(df['Complaint'])]


# In[14]:


#Create new df to hold only non-null values of Consumer complaint narratives.
df = df[pd.notnull(df['Complaint'])]


# In[15]:


df.shape


# In[11]:


stop_words = stopwords.words('english') + list(string.punctuation)
stop_words += ["''", '""', '...', '``', '--', 'xxxx']


# In[12]:


#Tokenize complaint data and remove stop words from complaint narrative.
def processComplaint(comp):
    tokens = nltk.word_tokenize(comp)
    removed_stop_words = [token.lower() for token in tokens if token.lower() not in stop_words]
    new_removed_stop_words = [word for word in removed_stop_words if word.isalpha()]
    
    return new_removed_stop_words


# In[13]:


#Link words together.
def linkWords(words):
    linked_words = ''
    
    for w in words:
        linked_words += w + ' '
    
    return linked_words.strip()


# In[14]:


lm = WordNetLemmatizer()


# In[15]:


#Group variants of the same word and merge complaints.
def groupVariants(words):
    words = [word for word in words if word is not np.nan]
    
    lem_list = []
    
    for idx, word in enumerate(words):
        lem_list.append(lm.lemmatize(word))
    
    linked_str = linkWords(lem_list)
    
    return linked_str


# In[16]:


nltk.download('wordnet')
nltk.download('omw-1.4')


# In[17]:


# Eliminate stop words and group variants of words.
for i in range(len(df)):
    processed_complaints = processComplaint(df['Complaint'].iloc[i])
    complaint = groupVariants(processed_complaints)
    
    df['Complaint'].iloc[i] = complaint

    #Keep track of processed complaints.
    if i % 5000 == 0:
        print(f'Processed Row: {i}')


# In[18]:


#Output processed complaints file.
df.to_csv('processed_complaints.csv')

