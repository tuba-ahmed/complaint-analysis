#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries and Preprocess Data

# In[1]:


import pandas as pd
import os
import json


# In[2]:


data = pd.read_csv("processed_complaints.csv")


# In[6]:


data = data.drop(['Unnamed: 0'],1)
data.head(5)


# In[7]:


data['Product'].value_counts()


# In[8]:


product_map = []
product_dict = {}
for i,prod in enumerate(data['Product'].unique()):
    product_map.append({"Product":prod, "id":i})
    product_dict[prod] = i


# In[9]:


product_dict


# ## PART 1: Predicting Subject

# In[10]:


data['output'] = data['Product'].map(product_dict)


# In[11]:


data.head()


# In[12]:


data = data[['Complaint','output']]


# In[13]:


data.head()


# In[20]:


data['Complaint'] = data['Complaint'].astype(str)


# In[14]:


get_ipython().system('pip install nltk')


# In[18]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
stop_words = stopwords.words('english')
STOPWORDS = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# In[21]:


MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['Complaint'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


# In[22]:


X = tokenizer.texts_to_sequences(data['Complaint'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# In[23]:


Y = pd.get_dummies(data['output']).values
print('Shape of label tensor:', Y.shape)


# In[24]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data['Complaint'],data['output'], test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ### Build Model

# In[25]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(30, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


epochs = 1
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss')])


# In[ ]:


model.save("Model")


# In[51]:





# In[ ]:




