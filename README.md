# Email Spam Classifier

This project is an **Email Spam Classifier** designed to detect and classify spam emails. It uses natural language processing techniques for feature extraction and Naive Bayes models for classification. The project demonstrates a full pipeline of text preprocessing, feature engineering, and machine learning model training to differentiate between legitimate and spam emails effectively.

## Features

1. **Feature Transformation**
   - **Tokenization**: Splits email content into individual tokens (words).
   - **Removing Special Characters, Stop Words, and Punctuation**: Cleans and standardizes the data by removing unnecessary elements, retaining only meaningful words.
   - **Stemming**: Reduces words to their root forms, enhancing the model's ability to generalize.

2. **Vectorization**
   - **Count Vectorizer**: Converts text data into a matrix of token counts.
   - **TF-IDF Vectorizer**: Uses Term Frequency-Inverse Document Frequency to give more weight to important words.

## Libraries Used

- **NLTK**: For text preprocessing, tokenization, stop word removal, and stemming.
- **Scikit-learn (sklearn)**: For machine learning models, vectorization, and performance evaluation.

## Models Used

Three Naive Bayes models are tested and compared for performance:
- **GaussianNB**
- **MultinomialNB**
- **BernoulliNB**

## Results
Precision focuses on reducing false positives which in this case is, legitimate (non-spam) emails incorrectly labeled as spam. False positives can lead to real emails being mistakenly sent to the spam folder. Hence we want maximum precision in this case. 
Out of all the models, the Multinomial Naive Bayes with TFIDF Vectorizer gives the best result with almost 100% Precision and 97% Accuracy.

