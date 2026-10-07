import os
from pathlib import Path
from urllib import response
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not api_key or not endpoint or not deployment:
    st.error("Missing Azure AI Foundry settings in .env")
    st.stop()

# Foundry uses the OpenAI-compatible endpoint
client = OpenAI(
    api_key=api_key,
    base_url=endpoint,
)

# Graph state definition
# @dataclass
class State(TypedDict):
    product_name: str
    basic_description: str
    features_benefits: str
    marketing_message: str
    final_description: str
    search_query: str
    justification: str

# Generate a basic product description
def generate_basic_description(state: State) -> dict:
    """Generate a basic description for the product."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates brief product descriptions."},
            {"role": "user", "content": f"Write a brief description of a product named '{state['product_name']}'."}
    ]
    )
    basic_description = response.choices[0].message.content
    return {"basic_description": basic_description}

# Add key features and benefits to the product description
def add_features_benefits(state: State) -> dict:
    """Add features and benefits to the product description."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": f"List key features and benefits of the product: {state['basic_description']}"}]
    )
    features_benefits = response.choices[0].message.content
    return {"features_benefits": features_benefits}

# Create a compelling marketing message based on the product's features
def create_marketing_message(state: State) -> dict:
    """Create a marketing message for the product."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": f"Create a compelling marketing message for the product: {state['features_benefits']}"}]
    )
    marketing_message = response.choices[0].message.content
    return {"marketing_message": marketing_message}

# Final polish and completion of the product description
def polish_final_description(state: State) -> dict:
    """Polish and finalize the product description."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": f"Polish and finalize the product description, incorporating the marketing message: {state['marketing_message']}"}]
    )
    final_description = response.choices[0].message.content
    return {"final_description": final_description}

# Function to build the workflow (Separate from Streamlit logic)
def build_workflow() -> StateGraph:
    """Build and compile the workflow steps using LangGraph."""

    # Build the workflow for generating the product description
    workflow = StateGraph(State)
    # Add nodes to the workflow (steps in the process)
    workflow.add_node("generate_basic_description", generate_basic_description) # Step 1: Basic Description
    workflow.add_node("add_features_benefits", add_features_benefits) # Step 2: Features and Benefits
    workflow.add_node("create_marketing_message", create_marketing_message) # Step 3: Marketing Message
    workflow.add_node("polish_final_description", polish_final_description) # Step 4: Final Description
    # Add edges to connect the nodes (steps in order)
    workflow.add_edge(START, "generate_basic_description")
    workflow.add_edge("generate_basic_description", "add_features_benefits")
    workflow.add_edge("add_features_benefits", "create_marketing_message")
    workflow.add_edge("create_marketing_message", "polish_final_description")
    workflow.add_edge("polish_final_description", END)
    # Compile the workflow into a chain of actions
    chain = workflow.compile()
    return chain

# Function to visualize the workflow (saved as an image)
def visualize_workflow():
    """Visualize and save the workflow as an image."""
    graph = nx.DiGraph()
    edges = [
        ("START", "generate_basic_description"),
        ("generate_basic_description", "add_features_benefits"),
        ("add_features_benefits", "create_marketing_message"),
        ("create_marketing_message", "polish_final_description")
    ]
    graph.add_edges_from(edges)
    plt.figure(figsize=(8, 5))
    nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
    plt.savefig("workflow.png")

# Main Streamlit function
def run_streamlit_app():
    """Handles the entire app logic: input, workflow, and output."""

    # Title for the app
    st.title("Product Description Generator")

    # Step 1: Take product name as input from the user
    product_name = st.text_input("Enter the product name:")# "Smart Water Bottle"

    # Step 2: Button to generate product description
    if st.button("Generate Product Description"):
        # Create the initial state with product name and empty fields for description steps
        state = State(product_name=product_name, basic_description="", features_benefits="", marketing_message="", final_description="")
        # Build and run the workflow
        chain = build_workflow()
        # Run the workflow and get the results
        result = chain.invoke(state)
        # Display the results in Streamlit
        st.subheader("Basic Description:")
        st.write(result["basic_description"])
        st.subheader("Features and Benefits:")
        st.write(result["features_benefits"])
        st.subheader("Marketing Message:")
        st.write(result["marketing_message"])
        st.subheader("Final Description:")
        st.write(result["final_description"])
        # Step 3: Visualize the workflow and save it as an image
        visualize_workflow()
        st.image("workflow.png", caption="Product Description Workflow")
if __name__ == "__main__":
    run_streamlit_app()