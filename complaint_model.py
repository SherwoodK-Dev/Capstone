import json
import re
from pathlib import Path

import faiss
import  nltk
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


def build_knowledge_base(knowledge_base_path):
    knowledge_base_path = Path(knowledge_base_path)

    with knowledge_base_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    kb_content = []
    for item in data:
        if item.get("doc_type") in {"resolution_guide", "sop"}:
            kb_content.append(str(item.get("content", "")).replace("\n", " "))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    kb_embeddings = embedding_model.encode(kb_content, show_progress_bar=False)
    kb_embeddings = np.asarray(kb_embeddings, dtype="float32")
    dimension = kb_embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(kb_embeddings)

    return {
        "embedding_model": embedding_model,
        "kb_content": kb_content,
        "index": index,
    }


def predict_category(user_text):
    embedding = model_value["embedding_model"].encode([user_text], show_progress_bar=False)
    embedding_scaled = model_value["scaler"].transform(embedding)
    prediction_index = np.argmax(model_value["ann_model"].predict(embedding_scaled, verbose=0), axis=1)[0]
    return model_value["label_encoder"].inverse_transform([prediction_index])[0]


def predict_priority(user_text):
    embedding = model_value["embedding_model"].encode([user_text], show_progress_bar=False)
    embedding_scaled = model_value["priority_scaler"].transform(embedding)
    prediction_index = np.argmax(model_value["priority_model"].predict(embedding_scaled, verbose=0), axis=1)[0]
    return model_value["priority_label_encoder"].inverse_transform([prediction_index])[0]


def query_knowledge_base(query_text):
    if not knowledge_bundle["kb_content"]:
        return ""

    k = 2
    query_embedding = knowledge_bundle["embedding_model"].encode([query_text], show_progress_bar=False).astype("float32")
    k = min(k, len(knowledge_bundle["kb_content"]))
    distances, indices = knowledge_bundle["index"].search(np.asarray(query_embedding).astype("float32"), k)
    retrieved = [knowledge_bundle["kb_content"][int(idx)] for idx in indices[0]]
    return retrieved[0] if len(retrieved) == 1 else retrieved


def train_and_prepare_model(csv_path, knowledge_base_path=None):
    global model_value
    model_value = {}
    df = pd.read_csv(csv_path)
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

    print(f"\nTraining Accuracy: {training_accuracy:.4f}")
    print(f"Testing Accuracy: {testing_accuracy:.4f}")
    print("\nANN Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

    priority_X_train, priority_X_test, priority_y_train, priority_y_test = train_test_split(
        X,
        df["Priority_Encoded"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["Priority_Encoded"].values,
    )
    priority_scaler = StandardScaler()
    priority_X_train_scaled = priority_scaler.fit_transform(priority_X_train)
    priority_X_test_scaled = priority_scaler.transform(priority_X_test)

    priority_X_train_smote, priority_y_train_smote = SMOTE(random_state=42).fit_resample(
        priority_X_train_scaled, priority_y_train
    )
    priority_model = create_ann_model(
        priority_X_train_smote.shape[1], len(np.unique(priority_y_train))
    )
    priority_model.fit(
        priority_X_train_smote,
        priority_y_train_smote,
        epochs=3,
        validation_split=0.2,
    )

    priority_predictions = np.argmax(
        priority_model.predict(priority_X_test_scaled, verbose=0), axis=1
    )
    print(f"\nPriority Testing Accuracy: {accuracy_score(priority_y_test, priority_predictions):.4f}")
    print("\nPriority Classification Report:")
    print(
        classification_report(
            priority_y_test,
            priority_predictions,
            target_names=priority_label_encoder.classes_,
        )
    )

    global knowledge_bundle
    knowledge_bundle = {}
    if knowledge_base_path is not None:
        knowledge_bundle = build_knowledge_base(knowledge_base_path)
    model_value = {
        "ann_model": ann_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "priority_model": priority_model,
        "priority_scaler": priority_scaler,
        "priority_label_encoder": priority_label_encoder,
        "embedding_model": embedding_model,
        "knowledge_base": knowledge_bundle,
    }
    return model_value

def get_model_value():
    return model_value
