import pickle
from src.preprocess import clean_text


# Load model once
with open("models/model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)


def predict_message(message):
    message = clean_text(message)
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"