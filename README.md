# Sentiment Analysis of Amazon Beauty Products Reviews

This project focuses on performing sentiment analysis on Amazon beauty product reviews using various machine learning models and natural language processing (NLP) tools. The goal is to classify product reviews into sentiment categories (positive, neutral, negative) and explore sentiment distributions based on various features such as verified purchase status, user activity, and product performance.

## Table of Contents
1. Overview

2. Data Collection
3. Data Preprocessing

4. Modeling
5. Evaluation
6. Insights and Analysis
7. Conclusion
8. Installation
9. License

# Overview
    
The project aims to perform sentiment classification on Amazon beauty product reviews. We used multiple machine learning models and sentiment analysis tools, including VADER and TextBlob, to predict the sentiment (positive, neutral, or negative) of each review. The project also includes additional analysis of product performance based on sentiment, verified purchase status, user engagement, and sentiment scores.

# Data Collection
The dataset used for this project includes Amazon reviews related to beauty products. The data contains the following columns:

rating_sentiment: Sentiment of the review (positive, neutral, negative)
verified_purchase: Whether the review was made by a verified purchaser
text: Review text
user_id: Unique identifier for each user
product_title: Title of the product
The dataset is processed and analyzed using Python libraries such as pandas, scikit-learn, XGBoost, VADER, and TextBlob.

# Data Preprocessing
The data preprocessing steps include:

1. Text Preprocessing: Tokenizing, removing stop words, and normalizing the text data.
2. Feature Extraction: Using TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction.
3. Sentiment Analysis: Applying VADER and TextBlob to calculate sentiment scores for each review.
4. Label Encoding: Encoding sentiment labels into numerical values for model training.

# VADER and TextBlob Sentiment Analysis
* VADER Sentiment: VADER (Valence Aware Dictionary and sEntiment Reasoner) is a tool for sentiment analysis that is specifically designed to analyze social media texts. It provides a sentiment score that can be used to classify reviews into positive, neutral, or negative sentiments.
* TextBlob Sentiment: TextBlob is a Python library for processing textual data. It provides a simple API for common NLP tasks, including sentiment analysis. It returns polarity and subjectivity scores, where polarity indicates sentiment (positive or negative), and subjectivity measures the degree of subjectivity of the text.

# Modeling
Multiple machine learning models were used for sentiment classification:

1. Random Forest Classifier: A decision tree-based model tuned with hyperparameter optimization using GridSearchCV.
2. Naive Bayes Classifier: A probabilistic model for text classification.
3. Support Vector Machine (SVM): A powerful classifier with a linear kernel for sentiment classification.
4. XGBoost: An optimized gradient boosting model for handling imbalanced data and improving prediction performance.
Each model was trained on resampled data to address class imbalances in the dataset. The models were evaluated based on accuracy, precision, recall, and F1-score for each sentiment class.

# Evaluation
Model evaluation metrics were calculated for each model, with the best model being selected for final predictions. The evaluation results for each model were as follows:

1. Random Forest Classifier: Achieved an accuracy of 90.91%, with strong performance for the positive class (precision: 0.96, recall: 0.96).
2. Naive Bayes Classifier: Achieved an accuracy of 82.58%, with decent performance for the positive class (precision: 0.95, recall: 0.88), but struggles with the negative and neutral classes.
3. Support Vector Machine (SVM): Achieved an accuracy of 87.55%, with excellent performance on the positive class (precision: 0.88, recall: 0.99), but struggled with the negative and neutral classes.
4. XGBoost: Achieved an accuracy of 89.26%, performing excellently on the positive class and neutral class, but showed lower performance for the negative class.
   
# Insights and Analysis
1. Sentiment Distribution by Verified Purchase: Verified purchasers tend to provide more reviews across all sentiment categories (positive, neutral, and negative), making their feedback more valuable.
2. User with the Most Reviews: A user with ID "AG73BVBKUOH22USSFJA5ZWL7AKXA" posted 165 reviews, showing a generally positive sentiment with an average sentiment score of 0.64.
3. Top 20 Products with Most Negative Sentiment: Products like "DMSO Cream with Aloe Vera" and "Collapsible Hair Diffuser" have the highest number of negative sentiment reviews, indicating potential areas for improvement in product quality.
   
# Conclusion
This sentiment analysis project provides valuable insights into customer sentiments about Amazon beauty products. The analysis identifies products with negative sentiment that could benefit from quality improvements. Additionally, verified purchasers tend to provide more useful feedback, and frequent reviewers can influence product reputation.

# Installation
To run this project on your local machine, you need to have the following dependencies installed:

1. Python 3.x
2. Required libraries:
3. pandas
4. numpy
5. matplotlib
6. seaborn
7. scikit-learn
8. xgboost
9. nltk
10. vaderSentiment
11. textblob
