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
from langgraph.graph import StateGraph, START, END
import requests
import streamlit as st
from workflow import build_workflow

# Load environment variables
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


def visualize_graph():
    """Generates a visualization of the workflow graph."""
    graph = nx.DiGraph()
    edges = [
        ("START", "analyzer_agent"),
        ("analyzer_agent", "investigate_agent"),
        ("investigate_agent", "planner_agent"),
        ("planner_agent", "communicator_agent"),
        ("communicator_agent", "escalation_agent"),
        ("escalation_agent", "END")
    ]
    graph.add_edges_from(edges)
    
    plt.figure(figsize=(8, 5))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
    
    plt.savefig("workflowE2e.png")
    output_path = Path(__file__).resolve().parent / "workflowE2E.png"
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


if __name__ == "__main__":
    build_workflow()
    image_path = visualize_graph()
    st.image(image_path, caption="E2E Workflow")
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
            st.subheader("Complaint Analysis:")
            st.write(result["complaint_severity_msg"])
            st.subheader("Investigation Analysis:")
            st.write(result["knowledge_base_response"])
            st.subheader("Planner Analysis:")
            st.write(result["resolution_plan"])
            st.subheader("Communicator Analysis:")
            st.write(result["customer_response"])
            st.subheader("Escalation Analysis:")
            st.write(result["escalation_decision"])