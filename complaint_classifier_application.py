import os
from pathlib import Path
from urllib import response
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AzureOpenAI, OpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from workflow import build_workflow

import requests
import streamlit as st

load_dotenv(Path(__file__).resolve().parent / ".env")

DATA_PATH = Path(__file__).resolve().parent / "complaints_train.csv"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"


class State(TypedDict):
    complaint_prompt: str
    complaint_severity_msg: str
    knowledge_base_response: str = ""
    resolution_plan: str = ""
    customer_response: str = ""
    escalation_decision: str = ""

def run_streamlit_app():
    st.title("Complaint Entry")
    st.write("Please fill out a complaint if you have one.")

    complaint_prompt = st.text_area("Enter your complaint", height=150)

    if st.button("Generate"):
        state = State(complaint_prompt=complaint_prompt)
        if not complaint_prompt.strip():
            st.warning("Please enter a complaint first.")
        else:
            data = {"complaint_prompt": complaint_prompt}
            prediction_api_url = os.getenv(
                "PREDICTION_API_URL", "http://127.0.0.1:8000/predict/"
            )
            response = requests.post(prediction_api_url, json=data)
            response.raise_for_status()
            payload = response.json()
            workflow = build_workflow()
            result = workflow.invoke(state)
            st.subheader("Complaint Category:")
            st.write(payload["prediction"])
            st.subheader("Complaint Analysis:")
            st.write(result["complaint_severity_msg"])
            st.subheader("Escalation Decision:")
            st.write(result["escalation_decision"])


if __name__ == "__main__":
    run_streamlit_app()