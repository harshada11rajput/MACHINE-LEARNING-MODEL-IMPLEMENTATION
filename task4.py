import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a dictionary 
data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'message': ['Hello, how are you?', 'Congratulations! You\'ve won a free ticket.', 'What\'s up?', 'Get free cash now!', 'How was your day?', 'Limited time offer! Buy now.', 'I love you.', 'You are a winner!', 'What are you doing?', 'Free trial offer!']
}

# Create DataFrame
df = pd.DataFrame(data)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data 
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test the model
sample_msg = ["Congratulations! You've won a free ticket."]
sample_vec = vectorizer.transform(sample_msg)
print("Spam":", model.predict(sample_vec))
