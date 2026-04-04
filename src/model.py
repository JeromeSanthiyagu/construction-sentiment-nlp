import pandas as pd
import pickle
import nltk
import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize analyzers
vader = SentimentIntensityAnalyzer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Extract VADER + TextBlob features
def extract_sentiment_features(text):
    vader_score = vader.polarity_scores(text)['compound']

    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity

    return [vader_score, textblob_polarity, textblob_subjectivity]

def train_model():
    print("Loading data...")
    df = pd.read_csv('construction_notes.csv')

    print("Preprocessing data...")
    df['Cleaned_Note'] = df['Note'].apply(preprocess_text)

    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_features = tfidf.fit_transform(df['Cleaned_Note']).toarray()

    print("Extracting VADER + TextBlob features...")
    sentiment_features = np.array(
        df['Cleaned_Note'].apply(extract_sentiment_features).tolist()
    )

    # Combine TF-IDF + VADER + TextBlob
    X = np.hstack((tfidf_features, sentiment_features))
    y = df['Sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Decision Tree Classifier
    # -----------------------------
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)

    print("\nDecision Tree Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

  
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
    ocsvm.fit(X_train)

    anomaly_pred = ocsvm.predict(X_test)

    print("\nOCSVM Results")
    print("1 = normal, -1 = anomaly")
    print(anomaly_pred[:20])

    # Save everything
    with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(dt_model, f)

    with open('ocsvm_model.pkl', 'wb') as f:
        pickle.dump(ocsvm, f)

    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    print("\nModels saved successfully!")

if __name__ == "__main__":
    train_model()
