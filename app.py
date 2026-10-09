from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load the saved model
model = joblib.load('complaints_model.pkl')

app = FastAPI()

# Define the input data structure using Pydantic
class InputData(BaseModel):
    complaint_prompt: str

# Define the prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Debugging: print the incoming complaint prompt
    print("Received complaint prompt:", data.complaint_prompt)

    # Check if the complaint prompt is provided
    if not data.complaint_prompt:
        return {"error": "No complaint prompt provided"}

    # Convert the input complaint prompt into a numpy array and reshape it for the model
    features = np.array([data.complaint_prompt])  # Assuming the model expects a string input

    # Make the prediction
    prediction = model.predict(features)
    print(prediction)
    # Return the prediction result
    return {"prediction": int(prediction[0])}
