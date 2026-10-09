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

import streamlit as st

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")
# deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

# if not api_key or not endpoint or not deployment:
#     raise RuntimeError("Missing Azure OpenAI credentials in .env")


# client = OpenAI(
#     api_key=api_key,
#     base_url=endpoint,
# )
DATA_PATH = Path(__file__).resolve().parent / "complaints_train.csv"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

try:
    from agents import (
        analyzer_agent,
        investigate_agent,
        planner_agent,
        communicator_agent,
        escalation_agent,
    )
    from state import InquiryState
except ModuleNotFoundError:
    from agents import (
        analyzer_agent,
        investigate_agent,
        planner_agent,
        communicator_agent,
        escalation_agent,
    )
    from state import InquiryState



# Graph state definition
# @dataclass
class State(TypedDict):
    complaint_prompt: str
    complaint_severity_msg: str
    knowledge_base_response: str = ""
    resolution_plan: str = ""
    customer_response: str = ""
    escalation_decision: str = ""





def build_workflow() -> StateGraph:
    workflow = StateGraph(State)
    workflow.add_node("analyzer_agent", analyzer_agent)
    workflow.add_node("investigate_agent", investigate_agent)
    workflow.add_node("planner_agent", planner_agent)
    workflow.add_node("communicator_agent", communicator_agent)
    workflow.add_node("escalation_agent", escalation_agent)
    # Add edges to connect the nodes (steps in order)
    workflow.add_edge(START, "analyzer_agent")
    workflow.add_edge("analyzer_agent", "investigate_agent")
    workflow.add_edge("investigate_agent", "planner_agent")
    workflow.add_edge("planner_agent", "communicator_agent")
    workflow.add_edge("communicator_agent", "escalation_agent")
    workflow.add_edge("escalation_agent", END)
    # Compile the workflow into a chain of actions
    chain = workflow.compile()
    return chain


# Main Streamlit function
def run_streamlit_app():
    st.title("Complaint Entry")
    st.write("Please fill out a complaint if you have one.")

    # Section 1: Simple prompt
    complaint_prompt = st.text_area("Enter your complaint", height=150)

    if st.button("Generate"):
        state = State(complaint_prompt=complaint_prompt)
        if not complaint_prompt.strip():
            st.warning("Please enter a complaint first.")
        else:
            workflow = build_workflow()
            result = workflow.invoke(state)
            st.subheader("Basic Description:")
            st.write(result)




if __name__ == "__main__":
    run_streamlit_app()