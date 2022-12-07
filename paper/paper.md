---
title: Comsumer Complaints Classification
date: "November 30, 2022"
author: Ganesh Nehru (009509747), Anshul Kumar (016039894), Ramkumar Sivakumar (015443727), Tuba Ahmed (014534429)

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

Customer satisfaction plays a crucial role in any business. In fact, it is shown that a company that has a high level of customer satisfaction yields a higher retention rate of customers, as well as having a positive brand reputation, and overall having a greater lifetime value. Customer satisfaction also plays an important role in driving customer loyalty and boosts their acquisition, as well as reflecting on the business team’s performance. With the digitization of most businesses, managing customer satisfaction in the digital space has become exceedingly important. While some industries have a rigorous focus on customer outcomes through government regularization, other businesses will benefit from introducing this aspect of process improvement into their workflow.

# Introduction

In this project, we will be constructing classification models of consumer complaints based on text mining and natural language processing techniques to produce insights into consumer behavior, which may benefit businesses into taking actionable steps to improve their customer satisfaction and retention. Our approach to this solution is two-fold. First, we aim to use standard sentiment analysis techniques to classify the complaints into standard categories. Second, we plan to apply topic modeling using LDA (Latent Dirichlet Allocation) to classify the given complaints and identify potential new categories. 

# Data Exploration and Preprocessing

### Feature Identification:
From initial observation, the dataset had eighteen feature columns. However, most of the features were redundant for our scenario, therefore, only a few features are collected from the consumers while submitting a complaint, such as: 
* Product
* Consumer complaint narrative
* Issues
* Sub-issues (not including personal information)

The consumers only provides a complaint narrative about 55% of the time with predefined issues listed in the form. “Submitted-via” is a feature that records the medium used by the consumer to file a complaint. “Submitted-via” is a feature that records the medium used by the consumer to file a complaint. This feature has seven unique values: web, email, referral, phone, web referral, fax, and mail post. Complaints from different mediums seem to have different volumes of complaints about different products. Hence, using the “Submitted-via” feature alongside the “consumer complaint narrative” could boost the F1 score and classification accuracy.


### Complaint Narrative Analysis:
This analysis answers whether the model can learn the input and target features. First, we preprocessed the complaint narratives by removing stop words and symbols. Later in the analysis, we extracted frequent word counts for complaint narratives of different product types. Then, the word-cloud function generated a word cloud for each product type's top 20 frequent words. The generated word cloud depicted a significant difference between the top 20 significant words used for different products, showing that the data is learnable and learning is feasible.

# Methods

### LSTM - Long Short Term Memory Network: 

The LSTM network was built using the NLTK and Keras library. The complaint narratives were tokenized and padded to a maximum sequence length of  50. The dataset was split into train and test sets with an 80-20 split. During training, 10% of the test set was used for validation. The estimated time for the LSTM model to finish one epoch was over an hour. Therefore CuDNNLSTM was used to speed up the training.  The model was trained for 10 epochs for 10 minutes with the default settings of Adam optimizer. Figure 3.1.1 compares Ein to Eout. The validation loss did not improve as the model started to overfit after 2 epochs. 

The performance of the trained model was evaluated on the test dataset. An F1 score of 0.86 with a classification accuracy of  86% was obtained on the test dataset (Figure 3.1.2). The evaluation metrics were poor for category 6 due to insufficient training samples. Hence, learning is not feasible for the category “Crypto Currency.”


### Multinomial Naive Bayes

Initially, We started off with preprocessing the data, where we removed redundant features, which resulted in a new data frame consisting of only the products and customer complaint narratives. Upon initial observation, we noticed that there were numerous products with similar titles. To rectify this issue, we relabeled the products and merged alike products together. This condensed the overall number of products from eighteen to seven products. Upon further analysis, we noticed that there was a total of 3096756 datapoints; of which, 1984336 were null values. These values were dropped, which resulted in a condensed dataset consisting of 1112420 datapoints. Finally, to complete the preprocessing step, we removed stop words, as well as tokenize and lemmatize the complaint narratives. This process took nearly five hours to complete. Upon completion, we exported the preprocessed complaints dataset, which we will be using for our modeling.

After the initial preprocessing, we created a new data frame from the preprocessed complaint file. Upon initial inspection of the data, we discovered that there were 81 null values in the complaint narratives, which we had to drop. The remaining data frame consisted of 1112339 datapoints. Next, we encoded the products in numeric values. We then proceeded to instantiate the training and test sets. Thirdly, we used TF-IDF vectorizer to obtain the relevancy of words in the complaint narratives, as mentioned in this commit. Finally, we created our model using the Multinominal Naïve Bayes classifier and produced a classification report, which yielded a prediction accuracy of 82%.

### SVM

Before the training of SVM and Logistic Regression model, the training dataset was obtained after processing the 'narrative' column for the dataset. As a pre-processing technique, stopwords were initially removed from each entry, which included strings like 'xx/xx/xxxx', puncuations and numericals. This was done to ensure that there is no noise in the training data. Since SVM runs slow, especially for large datasets, 100,000 samples were randonly selected from the dataset. Then the training data was split into 80% train and 20% test for validation purposes. 

To vectorize the text, tfidf was performed on the training and test set. Also for better performance for SVM Model, Singular Value Decomposition was performed on the training and test datasets, with 250 components (usually recommended for SVM). Since SVM only takes normalized values, the training and test dataset was scaled using StandardScaler. With this, SVC was trained with C=1.0 with probability estimates set to True.

Validation was performed on the test set created earlier, and F1 scoring metric was used to determine how well the model performed. For the trining set, the SVM model obtained a score of 0.87099 on the validation set. 

### K-Nearest Neighbors
To begin with the KNN implementation, there were some pre-processing steps that were needed to be done on the text data. We began by building a count vectorizer responsible for removing stop-words, cleaning and tokenizing the textual data. Next step in the process was to do a TF-IDF transformation. Before this, we cut down the data set to only components that we would be interested in – the product and complaint narratives.

Once the pre-processing was done and the TF-IDF transformation was complete, we proceeded to encode the product categories into numerical values and build a KNNClassifier model on top of this. We used sklearn library’s implementation for the same. As a group we decided to limit the sample size to 100,000. We finished up the implementation by adding the classification report. To enhance performance, we tried implementing the model using the RAPIDS accelerator. While this showed considerable improvement in the processing time, we decided to limit the sample size to standardize across implementations and minimize the performance factor and varying machine resources.

# Comparisons

| Method      | Precision | Recall     | F1 Score | Accuracy |
| :---        |    ----   |   ----     |   ---    | ---      |
| LSTM        | 0.72      | 0.85       | 0.73     | 0.86     |
| SVD         | 0.72      | 0.70       | 0.71     | 0.86     |
| Naive Bayes | 0.72      | 0.69       | 0.70     | 0.85     |
| KNN         | 0.73      | 0.72       | 0.70     | 0.72     |

The precision, recall, and F1 scores in table 4.1 are macro averages. The LSTM model had a better Recall, F1, and Accuracy score over the other models. KNN had a better precision score. Moreover, the time to train the LSTM model over ten epochs was around 10 minutes. Therefore, though SVM had the same accuracy as LSTM, the latter performed better with respect to Recall, F1, Accuracy, and Efficiency. But considering all the metrics, if we look at all of the models LSTM, apart from Precision, all the other metrics are relatively higher than the other models. So, with this, we selected the LSTM as the best-performing model for our classification problem.


# Conclusions

In the end, we were able to build models that classify the complaints with an accuracy of 86% and help streamline the consumer complaint filing process. We overcame the target variable ambiguity by renaming certain categories from the information we inferred in the data exploration steps. However, we did not have enough data to correctly classify complaint narratives of type “Crypt/Virtual Currency.”  
Our future work would be focused on improving classification accuracy using novel methods and additional features. During data exploration, we identified that the feature “Submitted-via”, the mode of submission of complaints by a consumer, had some influence on the target variable. Hence, the accuracy can be improved by incorporating more such features. 

