# -*- coding: utf-8 -*-
"""
Fake News Detection Project

This project involves building a machine learning model to distinguish between fake and true news articles. The data used is preprocessed, and features are extracted using TF-IDF. A Random Forest classifier is employed to classify the articles. Additionally, the project includes a time series analysis of article trends over time.
"""

# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# File paths
fake_news_path = '/content/drive/MyDrive/fakenewsdetection/fakenews.zip (Unzipped Files)/fake.csv'
true_news_path = '/content/drive/MyDrive/fakenewsdetection/fakenews.zip (Unzipped Files)/true.csv'

# Read datasets
fake_news_df = pd.read_csv(fake_news_path)
true_news_df = pd.read_csv(true_news_path)

# Add labels
fake_news_df['label'] = 0
true_news_df['label'] = 1

# Combine datasets
combined_df = pd.concat([fake_news_df, true_news_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in words])

combined_df['cleaned_text'] = combined_df['text'].apply(preprocess_text)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(combined_df['cleaned_text']).toarray()
y = combined_df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate model
y_pred = rf_classifier.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(rf_classifier, 'fake_news_detector_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Time series analysis
true_news_df['date'] = pd.to_datetime(true_news_df['date'], errors='coerce')
fake_news_df['date'] = pd.to_datetime(fake_news_df['date'], errors='coerce')
true_news_df = true_news_df.dropna(subset=['date'])
fake_news_df = fake_news_df.dropna(subset=['date'])
combined_df = pd.concat([true_news_df, fake_news_df], ignore_index=True)
combined_df['year_month'] = combined_df['date'].dt.to_period('M')

grouped_data = combined_df.groupby(['year_month', 'label']).size().reset_index(name='count')
time_series_data = grouped_data.pivot(index='year_month', columns='label', values='count').fillna(0)
time_series_data.columns = ['Fake_News', 'True_News']
time_series_data.index = time_series_data.index.to_timestamp()

# Plot trends
plt.figure(figsize=(14, 8))
plt.plot(time_series_data['Fake_News'], label="Fake News", linewidth=2)
plt.plot(time_series_data['True_News'], label="True News", linewidth=2)
plt.title("Time Series of Fake and True News Articles", fontsize=16, fontweight='bold')
plt.xlabel("Year-Month", fontsize=14)
plt.ylabel("Number of Articles", fontsize=14)
plt.grid(alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.show()


