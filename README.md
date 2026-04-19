# Spam Detection Project

## Overview
This project uses Machine Learning to classify messages as Spam or Not Spam.

## Features
- Text preprocessing
- TF-IDF vectorization
- Logistic Regression model
- CLI-based prediction

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python src/train.py

### 3. Run app
python main.py

#  Spam Detection System (ML Project)

 ## Overview

This project is a Machine Learning-based Spam Detection system that classifies messages as **Spam** or **Not Spam** using Natural Language Processing (NLP).

It includes:

* Data preprocessing
* TF-IDF feature extraction
* Logistic Regression model
* Model saving & loading
* CLI-based prediction system
* Data visualization (EDA)

##  Installation

1. Clone the repository:

git clone https://github.com/Pavankumarr25/spam-detection.git
cd spam-detection

2. Create virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

##  Model Training

Run the training script:

python -m src.train

This will:

* Train the model
* Evaluate performance
* Save model to `models/model.pkl`

##  Data Visualization

Generate graphs:

python -m src.visualize

Graphs will be saved in the `graphs/` folder.

##  Run the Application

python -m src.main

Example:

Enter message: Free lottery win!!!
Prediction: Spam


##  Model Performance

* Accuracy: ~94%
* F1 Score: ~0.74

## Features

* Text cleaning (regex + preprocessing)
* TF-IDF vectorization
* Logistic Regression classifier
* Confusion matrix & visual insights
* Modular and scalable project structure

## 🚀 Future Improvements

* FastAPI backend for real-time predictions
* Web UI (Streamlit / React)
* Model comparison (Naive Bayes, SVM)
* Deployment (Render / Railway / AWS)

##  Author

Pavan Kumar

---

## 📌 Note

Make sure to run training before prediction:

python -m src.train

