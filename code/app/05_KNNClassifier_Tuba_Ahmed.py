#!/usr/bin/env python
# coding: utf-8

# In[5]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


import pandas as pd

df =pd.read_csv("drive/MyDrive/CMPE-257-Project/complaints.csv")
df.head()


# Importing all required libraries

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# Restricting the columns to only the Product(Class) and the Complaint text.

# In[ ]:


df = df[['Product' , 'Consumer complaint narrative']]
df = df.rename(columns={'Product' : "Product",
                       'Consumer complaint narrative' : "Complaint"})
df.head()


# Cleaning up the Product column data.

# In[ ]:


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


# Removing the NaN values. For this project, we are only concerned with complaint text.

# In[ ]:


df = df[pd.notnull(df['Complaint'])]
df.head()


# Building a CountVectorizer - This includes Text pre-processing, tokenizing and filtering of stopwords.

# In[ ]:


stop_words = stopwords.words('english') + list(string.punctuation)
stop_words += ["''", '""', '...', '``', '--', 'xxxx']


# In[ ]:


#Tokenize complaint data and remove stop words from complaint narrative.
def processComplaint(comp):
    tokens = nltk.word_tokenize(comp)
    removed_stop_words = [token.lower() for token in tokens if token.lower() not in stop_words]
    new_removed_stop_words = [word for word in removed_stop_words if word.isalpha()]
    
    return new_removed_stop_words


# In[ ]:


#Link words together.
def linkWords(words):
    linked_words = ''
    
    for w in words:
        linked_words += w + ' '
    
    return linked_words.strip()


# In[ ]:


lm = WordNetLemmatizer()


# In[ ]:


#Group variants of the same word and merge complaints.
def groupVariants(words):
    words = [word for word in words if word is not np.nan]
    
    lem_list = []
    
    for idx, word in enumerate(words):
        lem_list.append(lm.lemmatize(word))
    
    linked_str = linkWords(lem_list)
    
    return linked_str


# In[ ]:


nltk.download('wordnet')
nltk.download('omw-1.4')


# In[ ]:


# Eliminate stop words and group variants of words.
for i in range(len(df)):
    processed_complaints = processComplaint(df['Complaint'].iloc[i])
    complaint = groupVariants(processed_complaints)
    
    df['Complaint'].iloc[i] = complaint


# Now, we'll consider the pre-processed complaints for further analysis.

# In[7]:


import pandas as pd
ndf = pd.read_csv("drive/MyDrive/CMPE-257-Project/processed_complaints.csv")
ndf.head()


# In[8]:


ndf.drop(['Unnamed: 0'], axis=1, inplace=True)
#Eliminate all null values.
ndf = ndf.dropna()
ndf['Product'].unique()


# In[9]:


ndf['Product'].replace({'Debt Collection' : 0, 
                        'Credit Reporting and Services' : 1,
                        'Banking Services' : 2,
                        'Mortgages' : 3,
                        'Credit/Prepaid Cards' : 4,
                        'Loans' : 5,
                        'Crypto Currency' : 6}, inplace=True)


# In[15]:


#Create train and test sets.
sample_df = ndf.sample(100000, random_state=42)
x = sample_df['Complaint']
y = sample_df['Product']


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


# In[12]:


print("Training set size:" , x_train.shape)
print("Test set size:" , x_test.shape)


# Performing a Tf-IDF transformation.

# In[20]:


tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000000)
tfidf_x_train = tfidf.fit_transform(x_train)
tfidf_x_test = tfidf.transform(x_test)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=7)

clf = KNeighborsClassifier().fit(tfidf_x_train, y_train)


# In[22]:


#Predictions for training and testing set.
pred_y_train = clf.predict(tfidf_x_train)
pred_y_test = clf.predict(tfidf_x_test)

print('Training prediction accuracy: ', accuracy_score(y_train, pred_y_train))
print('Testing prediction accuracy: ', accuracy_score(y_test, pred_y_test))


# In[23]:


print('Classification Report for KNN (Training Data)\n', classification_report(y_train, pred_y_train))


# In[24]:


print('Classification Report for Naive Bayes (Test Data)\n', classification_report(y_test, pred_y_test))

