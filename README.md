# MACHINE-LEARNING-MODEL-IMPLEMENTATION

Company : Codtech IT solution

Name : Harshada Ashoksingh Rajput

Intern ID : CT04DL620

Domain : Python Programming

Duration : 4 weeks

Mentor : Neela Santhosh

*Description* : 
This project demonstrates the implementation of a predictive machine learning model using Scikit-learn to classify SMS messages as spam or ham (not spam). The model is built using the Naive Bayes algorithm, which is well-suited for text classification tasks.
1. Data Creation and Preprocessing
- A small dataset of 10 SMS messages is manually created.
- Messages are labeled as "ham" (0) or "spam" (1).
- A Pandas DataFrame is used to organize the data.
2. Text Vectorization
The SMS text is converted into numerical format using CountVectorizer, which transforms text into a matrix of word counts.
3. Train-Test Split
The data is split into 80% training and 20% testing sets using train_test_split.
4. Model Training
A Multinomial Naive Bayes model is trained on the vectorized data, which is effective for text classification problems.

*Output* : 
Accuracy: 1.0
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Confusion Matrix:
[[1 0]
 [0 1]]
Prediction: [1]
