from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

from text_preprocessing import preprocess

MODEL_PATH = Path(__file__).resolve().parent / "complaints_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

class InputData(BaseModel):
    complaint_prompt: str

@app.post("/predict/")
def predict(data: InputData):
    
    print("Received complaint prompt:", data.complaint_prompt)
    
    if not data.complaint_prompt:
        return {"error": "No complaint prompt provided"}

    cleaned_complaint = preprocess([data.complaint_prompt])
    embeddings = embedding_model.encode(cleaned_complaint)
    features = scaler.transform(np.asarray(embeddings))

    probabilities = model.predict(features, verbose=0)
    prediction = int(np.argmax(probabilities[0]))
    return {"prediction": label_encoder.inverse_transform([prediction])[0]}
