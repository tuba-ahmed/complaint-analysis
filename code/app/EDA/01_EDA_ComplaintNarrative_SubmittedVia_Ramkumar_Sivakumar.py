#!/usr/bin/env python
# coding: utf-8

# ## AIM:
# 
# * **To explore if the medium of submitting a complaint has any significance on the product type and if including this feature in bulding classification models could help improve the accuracy and F1 score.**
# * **To understand if the complaints with missing narratives has any inderlying patterns.**
# * **To analyse the most significant words of complaint narratives pertaining to different target type.**
# 

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


df = pd.read_csv("complaints.csv")
df.shape


# # 'Submitted via' FEATURE ANALYSIS:

# In[9]:


df['Submitted via'].value_counts()


# In[148]:


237225/3096756


# In[29]:


df[df['Submitted via'] == "Web"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# In[30]:


df[df['Submitted via'] == "Referral"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# In[32]:


df[df['Submitted via'] == "Phone"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# In[33]:


df[df['Submitted via'] == "Postal mail"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# In[34]:


df[df['Submitted via'] == "Fax"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# In[35]:


df[df['Submitted via'] == "Web Referral"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# In[36]:


df[df['Submitted via'] == "Email"].groupby('Product').agg({'Product':'size'}).plot.barh(y='Product')


# ### SUMMARY:
# 
# * About 83% of the complaints are submitted via "Web". Whereas other mediums such as "Referral, Fax, Phone, Postal Mail, Email" accounts for only about 17%.
# * Web: Most of the complaints are for Credit Reporting and Services.
# * Referral: Most of the complaints are for Mortgages.
# * Phone, Postal Mail, and Fax: Most of the complaints are for Credit Reporting and Mortgages.
# * Web Referral: Most of the complaints are related checkings and savings account.
# * Email: Most of the complaints are regarding mortgages and bank services.

# ## COMPLAINT NARRATIVE MISSING VALUE ANALYSIS

# In[42]:


df.groupby("Product").agg({'Consumer complaint narrative':'size'}).plot.barh()


# In[45]:


df[df['Consumer complaint narrative'].isna()]['Product'].value_counts()


# In[46]:


df[df['Consumer complaint narrative'].isna()]['Submitted via'].value_counts()


# In[56]:


a = df[df['Consumer complaint narrative'].isna()].groupby('Product').agg({'Product':'size'}).rename(columns={'Product':'Missing'}).reset_index()


# In[58]:


b = df.groupby('Product').agg({'Product':'size'}).rename(columns={'Product':'Total'}).reset_index()


# In[60]:


missing_df = b.merge(a)


# In[61]:


missing_df['Percentage'] = missing_df['Missing']/missing_df['Total']


# In[62]:


missing_df


# In[65]:


sns.barplot(data=missing_df, x="Percentage", y="Product")


# In[68]:


df[~df['Consumer complaint narrative'].isna()].groupby('Product').agg({'Product':'size'}).rename(columns={'Product':'Total'}).reset_index()


# ### Summary:
# 
# * Bank Account type has the highest missing values of 82%.
# * All product types have missing consumer narrative at least by 45% except for product type - virtual currency. 
# * It can be inferred that only about half of the consumers chose to provide a complaint narrative. This could also siginify the importance of a complaint. 

# ## CONSUMER COMPLAINT - WORD ANALYIS on Preprocessed Data

# In[1]:


from tqdm import tqdm
import pandas as pd
from wordcloud import WordCloud
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv("processed_complaints.csv")
df.shape


# In[28]:


products = df['Product'].unique()
products


# In[123]:


class WordAnalysis:
    
    def __init__(self, df):
        
        self.product = []
        self.vocab_df = pd.DataFrame()
        self.df = df
        

    def getWords(self, product = 'Banking Services', top = 20):
        
        if self.vocab_df.empty == False:
            return self.vocab_df.head(top)
        vocab = {}
        words = []

        for x in tqdm(self.df[self.df['Product'] == product]['Complaint'].values):

            try:
                x = x.split(" ")
            except:
                continue

            for i in range(len(x)):

                if x[i] in vocab:
                    vocab[x[i]] += 1
                else:
                    vocab[x[i]] = 1

        vocab_df = pd.DataFrame(vocab.items(), columns=['Words', 'Count'])
        vocab_df = vocab_df.sort_values(by=['Count'], ascending=False).reset_index()
        self.vocab_df = vocab_df.drop(['index'],axis=1)
        self.product = product

        return self.vocab_df.head(top)

    def wordCloud(self, top=20):
        
        if self.vocab_df.empty:
            print("Run getWords(product, top) first.")
            return
        vocab_df = self.vocab_df.head(top)
        words = []
        m = vocab_df['Count'].min()
        for i, row in enumerate(vocab_df.values):
            row_words = [((row[0] + " "))] * int(row[1]/m)
            words+=row_words

        word_cloud = WordCloud(collocations = False, background_color = 'white').generate("".join(words))
        # Display the generated Word Cloud
        plt.imshow(word_cloud)
        plt.title(self.product)
        plt.axis("off")
        plt.show()
    


# In[130]:


m = WordAnalysis(df)


# In[131]:


m.getWords(products[0],20)


# In[135]:


m.wordCloud(20)


# In[136]:


m = WordAnalysis(df)
m.getWords(products[1],20)


# In[137]:


m.wordCloud(20)


# In[138]:


m = WordAnalysis(df)
m.getWords(products[3],20)


# In[139]:


m.wordCloud(20)


# In[140]:


m = WordAnalysis(df)
m.getWords(products[4],20)


# In[141]:


m.wordCloud(20)


# In[142]:


m = WordAnalysis(df)
m.getWords(products[5],20)


# In[143]:


m.wordCloud(20)


# In[144]:


m = WordAnalysis(df)
m.getWords(products[6],20)


# In[145]:


m.wordCloud(20)


# ### Summary:
# 
# *The list of preprocessed Product types:*
# 1. 'Debt Collection'
# 2. 'Credit Reporting and Services'
# 3. 'Banking Services', 'Mortgages'
# 4. 'Credit/Prepaid Cards'
# 5. 'Loans'
# 6. 'Crypto Currency'
# 
# The most frequently occured words in the complaint narratives differ per product types. The worldcloud images highligh the most frequent words pertaining to a particular product type.

# In[ ]:




