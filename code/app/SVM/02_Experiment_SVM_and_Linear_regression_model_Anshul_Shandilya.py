#!/usr/bin/env python
# coding: utf-8

# ##### **Our Project**
# 
# For our project, we decided to present 4 models our our text classification problem. The end goal for our project is to compare all four models and pick which one performed the best. 
# 
# For this project, I picked SVM to perform on the dataset. 
# 
# Scoring metric to be used is F1 score.

# #### **Goals for this notebook:**
# 
# * Perform cleaning and prepping of data
# * Choose a metric for scoring
# * Go through SVM model for text classification

# In[34]:


import pandas as pd
import string
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
stop_words = stopwords.words('english')


# In[3]:


# Loading the processed dataset created in the '02_Data_Exploration_Anshul_Shandilya' notebook.
df = pd.read_csv('data/cleaned_complaints.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[16]:


df.loc[345]['narrative']


# Here, we still have the narrative text in normal text. Let's see if we can do something about it.

# In[36]:


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[19]:


stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
stopwords_list += ["XXXX", "xxxx"]


# In[49]:


import re

def process_text(text):
  '''
    Function to process the text and return a list of words with stopwords and punctuations removed
  '''
  tokens = word_tokenize(text)

  # Revove tokens with stop words removed.
  ret_tokens_st = [token.lower() for token in tokens if token.lower() not in stopwords_list]

  # Remove tokens with 2 or more consecutive x's
  for token in ret_tokens_st:
    if re.search(r'x{2,}', token):
        ret_tokens_st.remove(token)

  # Remove tokens with string that contains two or more consecutive X's
  for token in ret_tokens_st:
    if re.search(r'X{2,}', token):
        ret_tokens_st.remove(token)


  # Remove tokens with 2 or more consecutive -'s
  for token in ret_tokens_st:
    if re.search(r'-{2,}', token):
        ret_tokens_st.remove(token)

  # Remove tokens with 2 or more consecutive .'s
  for token in ret_tokens_st:
    if re.search(r'\.{2,}', token):
        ret_tokens_st.remove(token)

  # Remove tokens with float numbers
  for token in ret_tokens_st:
    if re.search(r'\d+\.\d+', token):
        ret_tokens_st.remove(token)

  # Remove tokens with date in format xx/xx/xxxx
  for token in ret_tokens_st:
    if re.search(r'\d+/\d+/\d+', token):
        ret_tokens_st.remove(token)
      
        
  # Remove tokens with numbers  
  ret_tokens = [token for token in ret_tokens_st if not token.isnumeric()]

  return ret_tokens

# function to concat words (used in function below)
def concat_words(list_of_words):
    # remove any NaN's
    # list_of_words = [i for i in list if i is not np.nan]

    concat_words = ''
    for word in list_of_words:
        concat_words += word + ' '
    return concat_words.strip()

def perform_lemmatization(text):
  '''
      Function to perform lemmatization on the text and return concatenated string of lemmatized words separated by space.
  '''
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
  return ' '.join(lemmatized_words)

  # # lemmatize each word
  # lemmatizer = WordNetLemmatizer()
  # lemmatized_list = []
  # for idx, word in enumerate(text):
  #     lemmatized_list.append(lemmatizer.lemmatize(word))
  
  # # make the list into a single string with the words separated by ' '
  # lemmatized_text = concat_words(lemmatized_list)


  # return lemmatized_text


# In[24]:


len(df)


# Since working on this dataset, the dataset is huge. So we will work only on randomly selected sample of 10000 from the original dataset.

# In[70]:


sample_size = 100000
sample_df = df.sample(sample_size, random_state=42)
sample_df.reset_index(inplace=True)


# In[71]:


sample_df.head()
print(sample_df.shape)


# In[72]:


for i in range(sample_size):
    processed = process_text(sample_df['narrative'].loc[i])
    processed_lemm = perform_lemmatization(processed)
    sample_df['narrative'].loc[i] = processed_lemm

sample_df.head()


# ##### Saving the 10000 sample dataset.
# 
# Now that we have the processed complaints with processed narrative, we will save it

# In[73]:


sample_df.to_csv('data/processed_sample_df.csv', index=False)


# Now, we have loaded the dataset. Probably 'sub_product', 'Issue' and 'sub_issue' rows are redundant for our initial work as I only plan to use the 'product' as the class labels and the 'narrative' as the training data. 
# 
# ##### Initial Steps using the SVC algorithm:
# 
# * Extract the labels using preprocessing.LabelEncoder()
# * Split the data using the train_test_split() function into 20% test size and the rest as training data. Shuffle will we set as True to randomise the data
# * Use the 'narrative' row as our training data
# * Build the initial model

# Extracting the labels

# In[74]:


encoded_labels = preprocessing.LabelEncoder()
labels = encoded_labels.fit_transform(sample_df['product'])


# In[75]:


labels.shape


# Splitting the data into test and train data (20% test)

# In[76]:


# Splitting the dataset into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(sample_df['narrative'], labels, stratify = labels, test_size=0.2, random_state=47, shuffle=True)


# In[77]:


print(x_train.shape)
print(x_test.shape)


# ##### Now that we have our train and test data, we will build a model.

# First, I will initialise a TF-IDF model.

# In[78]:


tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')


# Now, I will apply the TF-IDF model on both the test and train data (will take time since the num. of data is high)

# In[79]:


tfidf.fit(list(x_train) + list(x_test))


# In[80]:


x_train_tfidf =  tfidf.transform(x_train)
x_test_tfidf = tfidf.transform(x_test)


# ##### **Training the SVM model**
# 
# 

# I learned that SVM model takes a lot of time to run. Especially with close to 90000 components, it probably won't end before at least a couple of hours. So need to reduce the number of components before I proceed.

# Reducing the number of components. (Using Singular Value Decomposition)
# 
# Initially, will try reducing to 300 components. 

# In[81]:


# Initializing and applying SVD on the TF-IDF vectorized data to reduce the num. of components to 250.
svd = TruncatedSVD(n_components=250)
svd.fit(x_train_tfidf)
x_train_svd = svd.transform(x_train_tfidf)
x_test_svd = svd.transform(x_test_tfidf)


# Now that we have the components reduced (using SVD), since SMV is a linear model, we need to normalize the data before we try to fit the data into SVM.

# In[82]:


# Normalizing the data using StandardScaler.
scl = preprocessing.StandardScaler()
scl.fit(x_train_svd)
x_train_svd_scl = scl.transform(x_train_svd)
x_test_svd_scl = scl.transform(x_test_svd)


# In[83]:


# Creating and traininng a SVC model
svm_model = SVC(C=1.0, probability=True)
svm_model.fit(x_train_svd_scl, y_train)
preds = svm_model.predict(x_test_svd_scl)


# In[84]:


# Printing the F-1 score
print("F1 score: ", metrics.f1_score(y_test, preds, average='weighted'))


# Also, I will try **logistic regression** below
# 

# One of the alternatives for TF-IDF is count vectorizer feature extraction. 

# In[85]:


# Initializing and training a count vector model
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words = 'english')
count_vect.fit(list(x_train) + list(x_test))
xtrain_count =  count_vect.transform(x_train)
xtest_count = count_vect.transform(x_test)


# In[86]:


# Save xtrain_count and xtest_count to disk for future use.
np.save('data/xtrain_count.npy', xtrain_count)
np.save('data/xtest_count.npy', xtest_count)


# In[87]:


# Initializing and training a logistic regression model
logistic_model = LogisticRegression(C=1.0)
logistic_model.fit(xtrain_count, y_train)
preds = logistic_model.predict(xtest_count)


# In[88]:


# Printing the F1 score
print("F1 score: ", metrics.f1_score(y_test, preds, average='weighted'))


# #### **Observations:**
# 
# * For sample_size=10000, two  models were trained:
# 
#     * **SVM** - F1 score = 0.8347134811772894
#     
#     * **Logistic Regression** - F1 score = 0.8049101124053899
# 
#     **For 10,000 samples taken from the dataset, SVM performed relatively better than Logistic Regression.** 
# 
# <br>
# 
# Since I took 10,000 samples, which is a fraction of close to million columns of data, I re-tried the same process with 100,000 samples.
# 
# * For sample_size=100,000, two  models were trained:
# 
#     * **SVM** - F1 score = 0.8550971142924789
#     
#     * **Logistic Regression** - F1 score = 0.854402883400011
# 
#     **But for 100,000 samples taken from the dataset, SVM still performed better than Logistic Regression but the differnece between them was relatively smaller.**
#     

# 
