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

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

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
    chain = workflow.compile()
    return chain

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
    output_path = Path(__file__).resolve().parent / "workflowE2e.png"
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


if __name__ == "__main__":
    build_workflow()
    image_path = visualize_graph()
    st.image(image_path, caption="E2E Workflow")