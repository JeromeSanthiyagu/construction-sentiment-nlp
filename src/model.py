import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization (split by space)
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv('construction_notes.csv')
    except FileNotFoundError:
        print("Error: construction_notes.csv not found. Run generate_data.py first.")
        return

    print("Preprocessing data...")
    df['Cleaned_Note'] = df['Note'].apply(preprocess_text)

    # Feature Extraction
    print("Extracting features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['Cleaned_Note']).toarray()
    y = df['Sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    print("Training model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    print("Saving model...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("Model saved to sentiment_model.pkl and tfidf_vectorizer.pkl")

if __name__ == "__main__":
    train_model()
