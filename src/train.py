import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import clean_text


def train_model():
    df = pd.read_csv("Data/spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].apply(clean_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, vectorizer), f)

    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    train_model()