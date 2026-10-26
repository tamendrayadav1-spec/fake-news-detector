"""
train_model.py
---------------
Trains the TF-IDF + Logistic Regression fake-news classifier and
saves the fitted vectorizer + model to the model/ folder so the
Flask app can load them instantly at request time (no retraining
per request).

Run:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    print("Loading dataset...")
    df = pd.read_csv("data/news_dataset.csv")
    df = df.dropna(subset=["text", "label"])

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        lowercase=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(model, "model/model.pkl")
    print("\nSaved model/vectorizer.pkl and model/model.pkl")


if __name__ == "__main__":
    main()
