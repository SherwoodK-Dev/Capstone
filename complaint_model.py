import json
import re
from pathlib import Path

import faiss
import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from imblearn.over_sampling import SMOTE
from string import punctuation

model_value = {}

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

DATA_PATH = Path(__file__).resolve().parent / "complaints_train.csv"


def preprocess(complaint_series):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned = []

    for complaint in complaint_series:
        text = str(complaint).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d+", "", text)
        words = word_tokenize(text)
        cleaned_words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word not in punctuation and word not in stop_words
        ]
        cleaned.append(" ".join(cleaned_words))

    return cleaned


def create_ann_model(input_dim, num_classes):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_and_prepare_model():
    df = pd.read_csv(DATA_PATH)
    df["timestamp_update"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp_update"].dt.hour        

    label_encoder = LabelEncoder()
    df["Category_Encoded"] = label_encoder.fit_transform(df["category"])
    priority_label_encoder = LabelEncoder()
    df["Priority_Encoded"] = priority_label_encoder.fit_transform(df["priority"])

    preprocessed = preprocess(df["complaint_text"])
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = embedding_model.encode(preprocessed)

    X = sentence_embeddings
    y = df["Category_Encoded"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    ann_model = create_ann_model(X_train_smote.shape[1], len(np.unique(y)))
    ann_model.fit(X_train_smote, y_train_smote, epochs=3, validation_split=0.2)

    predictions_proba = ann_model.predict(X_test_scaled, verbose=0)
    predictions = np.argmax(predictions_proba, axis=1)
    y_pred_train_proba = ann_model.predict(X_train_smote, verbose=0)
    y_pred_train = np.argmax(y_pred_train_proba, axis=1)

    training_accuracy = accuracy_score(y_train_smote, y_pred_train)
    testing_accuracy = accuracy_score(y_test, predictions)

    # print(f"\nTraining Accuracy: {training_accuracy:.4f}")
    # print(f"Testing Accuracy: {testing_accuracy:.4f}")
    # print("\nANN Classification Report:")
    # print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

    joblib.dump(ann_model, 'complaints_model.pkl')

if __name__ == "__main__":
    train_and_prepare_model()