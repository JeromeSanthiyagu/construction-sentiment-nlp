import pandas as pd
import pickle
import nltk
import re
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

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

    # Model Training - Naive Bayes for Sentiment
    print("Training Multinomial Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Evaluation
    print("Evaluating Sentiment model...")
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Model Training - One-Class SVM for Anomaly Detection
    # Using the entire dataset or just normal behavior? 
    # Usually OCSVM is trained on the entire feature set to find outliers.
    print("Training One-Class SVM for anomaly detection...")
    # Use a linear kernel for TF-IDF (high dimensional, sparse) and nu=0.01 to minimize false positives
    ocsvm_model = OneClassSVM(nu=0.01, kernel="linear")
    ocsvm_model.fit(X) # Fit on all data

    # Save models and vectorizer
    print("Saving models...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    with open('anomaly_model.pkl', 'wb') as f:
        pickle.dump(ocsvm_model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("Models saved to sentiment_model.pkl, anomaly_model.pkl, and tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_model()
