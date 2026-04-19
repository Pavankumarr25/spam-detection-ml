import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.preprocess import clean_text


def run_visualization():
    df = pd.read_csv("Data/spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["message"].apply(clean_text)

    # Graph 1: Bar
    label_counts = df["label"].value_counts()
    plt.figure()
    plt.bar(["Not Spam", "Spam"], label_counts.values)
    plt.title("Class Distribution")
    plt.savefig("graphs/bar.png")

    # Graph 2: Pie
    plt.figure()
    plt.pie(label_counts.values, labels=["Not Spam", "Spam"], autopct="%1.1f%%")
    plt.title("Spam vs Not Spam")
    plt.savefig("graphs/pie.png")

    # Graph 3: Histogram
    df["message_length"] = df["message"].apply(len)
    plt.figure()
    plt.hist(df["message_length"], bins=50)
    plt.title("Message Length Distribution")
    plt.savefig("graphs/hist.png")

    print("Graphs saved in /graphs folder")


if __name__ == "__main__":
    import os
    os.makedirs("graphs", exist_ok=True)
    run_visualization()