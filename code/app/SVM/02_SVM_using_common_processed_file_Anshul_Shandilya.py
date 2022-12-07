# %% [markdown]
# ## THIS NOTEBOOK USED FOR FINAL RESULTS FOR THIS MODEL. aLL SCREENSHOTS INCLUDED IN THE REPORT AND RESULTS PRESENT IN THIS NOTEBOOK

# %% [markdown]
# ##### **Our Project**
#
# For our project, we decided to present 4 models our our text classification problem. The end goal for our project is to compare all four models and pick which one performed the best.
#
# For this project, I picked SVM to perform on the dataset.
#
# Scoring metric to be used is F1 score.

# %% [markdown]
# #### **Goals for this notebook:**
#
# * Perform cleaning and prepping of data
# * Choose a metric for scoring
# * Go through SVM model for text classification

# %% [markdown]
# ## Note that I am not using the processed file that I have created in my previous notebooks. I am using the one processed by Ganesh.
#

# %%
# from google.colab import files
# uploaded = files.upload()


# %%

# unzip processed_complaints.zip -d processed_complaints.csv


# %%
import pandas as pd
import matplotlib.pyplot as plt
import string
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
# import nltk
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer
# stop_words = stopwords.words('english')

# %%
# Loading the processed dataset created in the '02_Data_Exploration_Anshul_Shandilya' notebook.
df = pd.read_csv('processed_complaints.csv')

# %%
df.shape

# %%
df.head()

# %%
# Change Complaint column to narrative
df = df.rename(columns={'Product': 'product', 'Complaint': 'narrative'})

# %%
df.head()

# %%
df.loc[345]['narrative']


# %% [markdown]
# Here, we still have the narrative text in normal text. Let's see if we can do something about it.

# %%
len(df)

# %% [markdown]
# Since working on this dataset, the dataset is huge. So we will work only on randomly selected sample from the original dataset.

# %%
sample_size = 100000
sample_df = df.sample(sample_size, random_state=42)
sample_df.reset_index(inplace=True)

# %%
sample_df.head()
print(sample_df.shape)

# %% [markdown]
# ##### Saving the sample dataset.
#
# Now that we have the processed complaints with processed narrative, we will save it

# %%
sample_df.shape

# %%
sample_df.isnull().sum()
sample_df.dropna(inplace=True)

# %% [markdown]
# Now, we have loaded the dataset. Probably 'sub_product', 'Issue' and 'sub_issue' rows are redundant for our initial work as I only plan to use the 'product' as the class labels and the 'narrative' as the training data.
#
# ##### Initial Steps using the SVC algorithm:
#
# * Extract the labels using preprocessing.LabelEncoder()
# * Split the data using the train_test_split() function into 20% test size and the rest as training data. Shuffle will we set as True to randomise the data
# * Use the 'narrative' row as our training data
# * Build the initial model

# %% [markdown]
# Extracting the labels

# %%
encoded_labels = preprocessing.LabelEncoder()
labels = encoded_labels.fit_transform(sample_df['product'])

# %%
labels.shape

# %% [markdown]
# Splitting the data into test and train data (20% test)

# %%
# Splitting the dataset into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(
    sample_df['narrative'], labels, stratify=labels, test_size=0.2, random_state=47, shuffle=True)

# %%
print(x_train.shape)
print(x_test.shape)

# %% [markdown]
# ##### Now that we have our train and test data, we will build a model.

# %%
# Function to plot the prediced values against the actual values


def plot_actual_vs_predicted(actual, predicted):
    figure, ax = plt.subplots()
    ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
    ax.plot([actual.min(), actual.max()], [
            actual.min(), actual.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.show()

# %% [markdown]
# First, I will initialise a TF-IDF model.


# %%
tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                        token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')

# %% [markdown]
# Now, I will apply the TF-IDF model on both the test and train data (will take time since the num. of data is high)

# %%
tfidf.fit(list(x_train) + list(x_test))

# %%
x_train_tfidf = tfidf.transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# %% [markdown]
# Comment either cell above or below depending on what you need to do.

# %% [markdown]
# ##### **Training the SVM model**
#
#

# %% [markdown]
# I learned that SVM model takes a lot of time to run. Especially with close to 90000 components, it probably won't end before at least a couple of hours. So need to reduce the number of components before I proceed.

# %% [markdown]
# Reducing the number of components. (Using Singular Value Decomposition)
#
# Initially, will try reducing to 300 components.

# %%
# Initializing and applying SVD on the TF-IDF vectorized data to reduce the num. of components to 250.
svd = TruncatedSVD(n_components=250)
svd.fit(x_train_tfidf)
x_train_svd = svd.transform(x_train_tfidf)
x_test_svd = svd.transform(x_test_tfidf)

# %% [markdown]
# Now that we have the components reduced (using SVD), since SMV is a linear model, we need to normalize the data before we try to fit the data into SVM.

# %%
# Normalizing the data using StandardScaler.
scl = preprocessing.StandardScaler()
scl.fit(x_train_svd)
x_train_svd_scl = scl.transform(x_train_svd)
x_test_svd_scl = scl.transform(x_test_svd)

# %%
# Creating and traininng a SVC model
svm_model = SVC(C=1.0, probability=True)
svm_model.fit(x_train_svd_scl, y_train)
preds = svm_model.predict(x_test_svd_scl)


# %%
preds_train = svm_model.predict(x_train_svd_scl)


# %%
# Printing the F-1 score
print("F1 score on train set: ", metrics.f1_score(
    y_train, preds_train, average='weighted'))
print("F1 score on test set: ", metrics.f1_score(
    y_test, preds, average='weighted'))


# %%
print("Classification report on train set: ")
print(metrics.classification_report(y_train, preds_train))

# %%
# Make a confusion matrix
cm = metrics.confusion_matrix(y_train, preds_train)
cm_display = metrics.ConfusionMatrixDisplay(cm).plot()
plt.show()

# %%
y_train.shape
y_test.shape

# %%
# # Load pickle model
# import pickle
# filename = 'svm_model.pkl'
# pickle.dump(svm_model, open(filename, 'wb'))


# %%
preds.shape

# %%
# Plot the predicted values against the actual values.
# plot_actual_vs_predicted(y_test, preds)

# %% [markdown]
# #### **Observations:**
#
# * For sample_size=10000,
#
#     * **SVM** - F1 score = 0.8347134811772894
#
# <br>
#
# Since I took 10,000 samples, which is a fraction of close to million columns of data, I re-tried the same process with 100,000 samples.
#
# * For sample_size=100,000,
#
#     * **SVM** - F1 score = 0.8550971142924789
#
#

# %% [markdown]
#
