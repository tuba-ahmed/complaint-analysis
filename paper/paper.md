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

# Methods

### Feature Identification:
From initial observation, the dataset had eighteen feature columns. However, most of the features were redundant for our scenario, therefore, only a few features are collected from the consumers while submitting a complaint, such as: 
* Product
* Consumer complaint narrative
* Issues
* Sub-issues (not including personal information)

The consumers only provides a complaint narrative about 55% of the time with predefined issues listed in the form. “Submitted-via” is a feature that records the medium used by the consumer to file a complaint. “Submitted-via” is a feature that records the medium used by the consumer to file a complaint. This feature has seven unique values: web, email, referral, phone, web referral, fax, and mail post. Complaints from different mediums seem to have different volumes of complaints about different products. Hence, using the “Submitted-via” feature alongside the “consumer complaint narrative” could boost the F1 score and classification accuracy.

### Complaint Narrative Analysis:
This analysis answers whether the model can learn the input and target features. First, we preprocessed the complaint narratives by removing stop words and symbols. Later in the analysis, we extracted frequent word counts for complaint narratives of different product types. Then, the word-cloud function generated a word cloud for each product type's top 20 frequent words. The generated word cloud depicted a significant difference between the top 20 significant words used for different products, showing that the data is learnable and learning is feasible.

### Multinomial Naive Bayes

Initially, We started off with preprocessing the data, where we removed redundant features, which resulted in a new data frame consisting of only the products and customer complaint narratives. Upon initial observation, we noticed that there were numerous products with similar titles. To rectify this issue, we relabeled the products and merged alike products together. This condensed the overall number of products from eighteen to seven products. Upon further analysis, we noticed that there was a total of 3096756 datapoints; of which, 1984336 were null values. These values were dropped, which resulted in a condensed dataset consisting of 1112420 datapoints. Finally, to complete the preprocessing step, we removed stop words, as well as tokenize and lemmatize the complaint narratives. This process took nearly five hours to complete. Upon completion, we exported the preprocessed complaints dataset, which we will be using for our modeling.

After the initial preprocessing, we created a new data frame from the preprocessed complaint file. Upon initial inspection of the data, we discovered that there were 81 null values in the complaint narratives, which we had to drop. The remaining data frame consisted of 1112339 datapoints. Next, we encoded the products in numeric values. We then proceeded to instantiate the training and test sets. Thirdly, we used TF-IDF vectorizer to obtain the relevancy of words in the complaint narratives, as mentioned in this commit. Finally, we created our model using the Multinominal Naïve Bayes classifier and produced a classification report, which yielded a prediction accuracy of 82%.



### K-Nearest Neighbors
To begin with the KNN implementation, there were some pre-processing steps that were needed to be done on the text data. We began by building a count vectorizer responsible for removing stop-words, cleaning and tokenizing the textual data. Next step in the process was to do a TF-IDF transformation. Before this, we cut down the data set to only components that we would be interested in – the product and complaint narratives.

Once the pre-processing was done and the TF-IDF transformation was complete, we proceeded to encode the product categories into numerical values and build a KNNClassifier model on top of this. We used sklearn library’s implementation for the same. As a group we decided to limit the sample size to 100,000. We finished up the implementation by adding the classification report. To enhance performance, we tried implementing the model using the RAPIDS accelerator. While this showed considerable improvement in the processing time, we decided to limit the sample size to standardize across implementations and minimize the performance factor and varying machine resources.


### Linear Regression

# Comparisons

# Example Analysis

# Conclusions


# References
