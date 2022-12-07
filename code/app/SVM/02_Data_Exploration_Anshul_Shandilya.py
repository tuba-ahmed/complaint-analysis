#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
import string


# In[2]:


def plotCountPlot(df, column, title, xlabel, ylabel, figsize=(10, 5), invert=False):
    '''
      Function to plot count plot for a given column in a dataframe
    '''
    plt.figure(figsize=figsize)
    if invert:
        sb.countplot(y=column, data=df)
    else:
      sb.countplot(x=column, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# ##### Loading the data

# In[3]:


# Reading the data
df = pd.read_csv('complaints.csv')


# In[4]:


# Initial investiogation of the data
df.head()


# In[5]:


print(df.shape)


# We read the data above. Now we rename the column names for our ease. 

# In[6]:


'''Get list of all column names'''
col_list = df.columns.tolist()
# print(col_list)

''' Renaming the columns for ease of access'''
df = df.rename(columns={'Date received':'date_received', 'Product':'product', 'Sub-product':'sub_product', 'Issue':'issue', 'Sub-issue':'sub_issue', 'Consumer complaint narrative':'narrative', 'Company public response':'response_public', 'Company':'company', 'State':'state', 'ZIP code':'zip', 'Tags':'tags', 'Consumer consent provided?':'consent', 'Submitted via':'submitted_via', 'Date sent to company':'date_sent', 'Company response to consumer':'response_to_cust', 'Timely response?':'timely_resp', 'Consumer disputed?':'consumer_disputed', 'Complaint ID':'id'})
df.head()


# Since the goal of this notebook is to see if we can perform product classification, we will now plot the product and sub-product columns to seee if that tells us something.

# In[7]:


# Plotting the countplot for 'product' column
sb.countplot(data=df, y='product')


# In[8]:


# Plotting the countplot for 'sub_product' column
sb_plot = sb.countplot(data=df, y='sub_product')
plt.figure(figsize=(4,8))
sb_plot.set_xticklabels(sb_plot.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# From the above plots, we can see the data is **highly imbalanced**.
# 
# * From the first graph, most of the data belongs to the "Credit reporting, credit repair services, or other personal consumer reports".
# * From the second graph, most of the data belongs to the "Credit reporting" class. 
# 
# The second graph only tells us that the data is imbalance. But there are too many sub_product classes for our problem. So product classes are more comprehensible. 
# 
# We might have to drop some of the classes that are not required for our problem or the ones that have null values.

# ##### Extracting relevant columns for our problem

# In[9]:


new_df = df[['product', 'sub_product', 'issue', 'sub_issue', 'narrative']]
new_df.head()


# In[10]:


plotCountPlot(new_df, 'product', 'Product Count', 'Product', 'Count', invert=True)


# In[11]:


# Getting the count of each product
new_df['product'].value_counts()


# In[12]:


# Getting the count of each sub product
new_df['sub_product'].value_counts()


# In[13]:


# Getting count of issue
new_df['issue'].value_counts()


# In[14]:


# Getting count of sub issue
new_df['sub_issue'].value_counts()


# ##### Analysis after seeing the counts:
# 
# There is surely a lot of data, but only 'product' seems to have enough data for training. All the other columns are highly imbalanced (even more than 'product') i.e. they have little to no data for some classes. 
# 
# Also for 'sub_product' there seems to be a large number of ambiguous data as there are classes like 'others', 'i don't know' and 'none'. If I decided to combine the 'product' and 'sub_product', then it might be a challenge as there would be overlaps.

# #### Assessing Null Values

# In[15]:


# List count of null values in each column
print("Count of null values in each column")
print(new_df.isnull().sum())
print("-----------------------")
# List count of all values minus null values in each column
print("Actual count of values in each column excluding null values")
print(new_df.count() - new_df.isnull().sum())


# In[16]:


# Find the percentage of null values in each column
print("Percentage of null values in each column")
print(new_df.isnull().sum() / new_df.shape[0] * 100)


# In[17]:


new_df.shape


# What can we make out of this null values?
# 
# Since we are doing text classification, we obviously need text to perform the classification. Mostly the text should have been in the narrative section. But we can see from the above, 'narrative' has 64% of values as NaN.
# 
# Since there are 3096756 rows in our data, we can comfortably remove about 64% of our data and still have significant data left for training. 

# ##### Dropping the rows where narrative is Null

# In[18]:


# Drop rows where narrative is null
new_df = new_df.dropna(subset=['narrative'])


# In[19]:


new_df.shape


# In[20]:


# Find the percentage of null values in each column
print("Percentage of null values in each column")
print(new_df.isnull().sum() / new_df.shape[0] * 100)


# We still have null values, but the percentage of them has dropped significantly. Now the dataset seems to be workable.

# ##### Working with the text data

# In[21]:


print(new_df.iloc[0])


# In[22]:


narrative_text = new_df['narrative'].iloc[0]
print(narrative_text)


# In[23]:


nltk.download('stopwords')
nltk.download('punkt')


# In[24]:


stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation)
stopwords_list += ["XXXX", "xxxx"]


# In[25]:


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

  # Remove tokens with date in format xx/xx/xxxxx
  for token in ret_tokens_st:
    if re.search(r'\d+/\d+/\d+', token):
        ret_tokens_st.remove(token)
        
  # Remove tokens with numbers  
  ret_tokens = [token for token in ret_tokens_st if not token.isnumeric()]

  return ret_tokens


# I defined a fucntion to remove the punctuaions and stop words from the string/text passed to it. 

# In[26]:


processed = process_text(narrative_text)
print(processed)


# Need to add stop words to stop list based on future finds of words like 'xxxx'

# Will try random rows to check if text needs to be filtered more
# 
# Things found to filter the text with:
# 
# * Consecutive x's
# * Consecutive -'s
# * Consecutive .'s
# * Numbers
# * Float numbers
# * Dates xx/xx/xxxx
# 
# Regex for the above have been added to the function to process the narrative.

# In[27]:


# Processed and tokenised words from the first narrative w/ 'xxxx' removed

narrative_text = new_df['narrative'].iloc[15405]

processed = process_text(narrative_text)
print(processed)

# Print the Frequency distribution of words in the first narrative
fdist = FreqDist(processed)

for each in fdist.most_common():
    print(each)


# In[28]:


fdist.plot()


# ##### Data manipulation (combining classes)

# Now that we have the baseline for text processing, we can move on to our previous question. 
# 
# Possibility 1:
# We have **'issue'**, **'sub_issue'**and **'narrative'** to work with in terms of data to perform classificaiton. So we will try to combine some data, maybe even combine these feautures.
# 
# Possibility 2: Try to merge **'products'** rows, as not all of them have satisfactory amount of data. 
# 

# In[29]:


# Getting the count of each product
new_df['product'].value_counts()


# We can look to see if we can merge/remove some of the rows for the 'product' column. 
# 
# We can definite see that some of the classes/categories seem to be the same. So we will try to look and them and see if we can merge them
# 

# In[30]:


# Get the head of the dataframe where product is 'Virtual Currency'
new_df[new_df['product'] == 'Virtual currency'].head(15)


# In[31]:


# Get the head of the dataframe where product is 'Money transfers, virtual currency, or money service'
new_df[new_df['product'] == 'Money transfer, virtual currency, or money service'].head(15)


# In[32]:


# Get the head of the dataframe where product is 'Money transfers'
new_df[new_df['product'] == 'Money transfers'].head(15)


# Now to check for all loans.

# In[33]:


# Get the head of the dataframe where product is 'Student loans'
new_df[new_df['product'] == 'Student loan'].head(15)


# In[34]:


# Get the head of the dataframe where product is 'Vehicle loan or lease'
new_df[new_df['product'] == 'Vehicle loan or lease'].head(15)


# In[35]:


# Get the head of the dataframe where product is 'Payday loan, title loan, or personal loan'
new_df[new_df['product'] == 'Payday loan, title loan, or personal loan'].head(15)


# All these three look to be the same or at least similar. So we can safely merge these three categeories and since 'virtual currencies' has very low data, it is better to merge it anyways.
# 

# In[36]:


new_df['product'].replace({'Virtual currency':'Money_and_virtual_transfers_and_services',
'Money transfer, virtual currency, or money service':'Money_and_virtual_transfers_and_services',
 'Money transfers': 'Money_and_virtual_transfers_and_services'}, inplace=True)


# In[37]:


new_df['product'].value_counts()


# We can safely merge the loans as there is not a lot of data in some of the 'xxxxx_loan' categeories. This would help in classifying as we combined and have more data for the 'loan' category.

# In[38]:


new_df['product'].replace({'Student loan':'loans',
'Vehicle loan or lease':'loans',
 'Payday loan, title loan, or personal loan': 'loans',
 'Consumer Loan':'loans',
 'Payday loan':'loans',}, inplace=True)


# In[39]:


new_df['product'].value_counts()


# Similarly, we can merge all similar to credit reporting ('Credit reporting, credit repair services, or other personal consumer reports' and 'Credit Reporting')

# In[40]:


new_df['product'].replace({'Credit reporting, credit repair services, or other personal consumer reports':'credit_reporting_and_services',
'Credit reporting':'credit_reporting_and_services'}, inplace=True)
new_df['product'].value_counts()


# Trying to figure out what 'Other financial service' are about...

# In[41]:


# Get the head of the dataframe where product is 'Other financial service'
new_df[new_df['product'] == 'Other financial service'].head(15)


# Since the columns 'Other financial service', 'Bank account or service', 'Checking or savings account' seem to bew similar, we will merge them

# In[42]:


new_df['product'].replace({'Other financial service':'banking_services',
'Bank account or service':'banking_services',
'Checking or savings account':'banking_services',}, inplace=True)
new_df['product'].value_counts()


# Since the columns 'Prepaid card', 'Credit card', 'Credit card or prepaid card' seem to bew similar, we will merge them

# In[43]:


new_df['product'].replace({'Prepaid card':'credit_card_or_prepaid_card',
'Credit card':'credit_card_or_prepaid_card',
'Credit card or prepaid card':'credit_card_or_prepaid_card'}, inplace=True)
new_df['product'].value_counts()


# In[44]:


new_df['product'].replace({'Debt collection':'debt_collection'}, inplace=True)
new_df['product'].value_counts()


# In[45]:


plotCountPlot(new_df, 'product', 'Product Count', 'Count', 'Product', invert=True)


# Data still seems to be imbalanced, but certainly better than before. Maybe I can try to marge more 'product' categories to be able to get decent data for classification.

# In[46]:


# Get the head of the dataframe where product is 'Mortgage'
new_df[new_df['product'] == 'Mortgage'].head(15)


# In[47]:


# Get the head of the dataframe where product is 'loans'
new_df[new_df['product'] == 'loans'].head(15)


# The categories 'loan' and 'mortgage' seem to be similar, but they are different enough to have their own categories. So, will not be merging them.
# 
# 
# But we can see that 'Money_and_virtual_transfers_and_services' has quite a low count, which would cause problems for training for classification. So it would be appropriate to merge them into 'banking_services'
# 

# In[48]:


new_df['product'].replace({'Money_and_virtual_transfers_and_services':'banking_services_and_transfers',
'banking_services':'banking_services_and_transfers'}, inplace=True)
new_df['product'].value_counts()


# In[49]:


plotCountPlot(new_df, 'product', 'Product Count', 'Count', 'Product', invert=True)


# We can see that the new data is more appropriate for classification problem because the mimimum count for a 'product' category is around 77000 and for data with more than a million entries, that is relatively enough for a classification problem, at least better than the 'product' classes  with close to 700 counts that we had before. 
# 
# So, at least at this point, we can assume that we have enough data to learn.

# In[50]:


# Get the head 
new_df.head(15)


# In[52]:


# Writing the data to csv file
new_df.to_csv('cleaned_complaints.csv', index=False)


# In[ ]:




