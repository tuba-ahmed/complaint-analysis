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

# Here, I'm going to try the TfidfVectorizer() instead of the process_text() function in the '02_Data_Exploration_Anshul_Shandilya' notebook.

# In[2]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[3]:


# Loading the processed dataset created in the '02_Data_Exploration_Anshul_Shandilya' notebook.
df = pd.read_csv('data/cleaned_complaints.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# Now, we have loaded the dataset. Probably 'sub_product', 'Issue' and 'sub_issue' rows are redundant for our initial work as I only plan to use the 'product' as the class labels and the 'narrative' as the training data. 
# 
# ##### Initial Steps using the SVC algorithm:
# 
# * Extract the labels using preprocessing.LabelEncoder()
# * Split the data using the train_test_split() function into 20% test size and the rest as training data. Shuffle will we set as True to randomise the data
# * Use the 'narrative' row as our training data
# * Build the initial model

# Extracting the labels

# In[6]:


encoded_labels = preprocessing.LabelEncoder()
labels = encoded_labels.fit_transform(df['product'])


# In[7]:


labels.shape


# Splitting the data into test and train data (20% test)

# In[8]:


# Splitting the dataset into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(df['narrative'], labels, stratify = labels, test_size=0.2, random_state=47, shuffle=True)


# In[9]:


print(x_train.shape)
print(x_test.shape)
#print(889936+222484)


# ##### Evaluation Metric

# Since we want to see how we perform, we will use multi-class log-loss as our evaluation metric. The code has been taken from "https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/util.py"

# In[ ]:


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


# ##### Now that we have our train and test data, we will build a model.

# First, I will initialise a TF-IDF model.

# In[9]:


tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')


# Now, I will apply the TF-IDF model on both the test and train data (will take time since the num. of data is high)

# In[10]:


tfidf.fit(list(x_train) + list(x_test))


# In[11]:


x_train_tfidf =  tfidf.transform(x_train)
x_test_tfidf = tfidf.transform(x_test)


# ##### **Training the SVM model**
# 
# 

# I learned that SVM model takes a lot of time to run. Especially with close to 90000 components, it probably won't end before at least a couple of hours. So need to reduce the number of components before I proceed.

# Reducing the number of components. (Using Singular Value Decomposition)
# 
# Initially, will try reducing to 300 components. 

# In[12]:


# Initializing and applying SVD on the TF-IDF vectorized data to reduce the num. of components to 250.
svd = TruncatedSVD(n_components=250)
svd.fit(x_train_tfidf)
x_train_svd = svd.transform(x_train_tfidf)
x_test_svd = svd.transform(x_test_tfidf)


# Now that we have the components reduced (using SVD), since SMV is a linear model, we need to normalize the data before we try to fit the data into SVM.

# In[1]:


# Normalizing the data using StandardScaler.
scl = preprocessing.StandardScaler()
scl.fit(x_train_svd)
x_train_svd_scl = scl.transform(x_train_svd)
x_test_svd_scl = scl.transform(x_test_svd)


# In[ ]:


# Creating and traininng a SVC model
svm_model = SVC(C=1.0, probability=True)
svm_model.fit(x_train_svd_scl, y_train)
preds = svm_model.predict(x_test_svd_scl)


# Since everything for SVM is taking a lot of time, will try **logistic regression**
# 

# One of the alternatives for TF-IDF is count vectorizer feature extraction. 

# In[10]:


# Initializing and training a count vector model
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words = 'english')
count_vect.fit(list(x_train) + list(x_test))
xtrain_count =  count_vect.transform(x_train)
xtest_count = count_vect.transform(x_test)


# In[12]:


# Save xtrain_count and xtest_count to disk for future use.
np.save('data/xtrain_count.npy', xtrain_count)
np.save('data/xtest_count.npy', xtest_count)


# In[13]:


# Initializing and training a logistic regression model
logistic_model = LogisticRegression(C=1.0)
logistic_model.fit(xtrain_count, y_train)
preds = logistic_model.predict(xtest_count)


# #### Here, we have encountered a problem, I do not have enough computing power to be able to train the models on the full approx. million columns of data. So, I will have to try reducing the data or even sampling from the data to perform our analysis.

# In[ ]:




