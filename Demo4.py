import streamlit as st
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import Literal
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path=r"C:\Users\kyles\Desktop\Agentic AI\Lesson_59\Lesson_59\.env")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not api_key or not endpoint or not deployment:
    st.error("Missing Azure AI Foundry settings in .env")
    st.stop()

# Foundry uses the OpenAI-compatible endpoint
client = openai.OpenAI(
    api_key=api_key,
    base_url=endpoint,
)

# Define state structure to track input, decision, and output
class State(TypedDict):
    input: str
    decision: str
    output: str

class Route(BaseModel):
    step: Literal["fantasy", "sci-fi", "mystery"] = Field(None, description="The next step in the routing process")

def get_router_response(input_text: str) -> str:
    """Uses AI model to categorize input into a specific genre."""
    response = client.chat.completions.create(model=deployment,
    messages=[
        {"role": "system", "content": "Route the input to 'fantasy', 'sci-fi', or 'mystery' based on its theme. If unsure, default to 'mystery'."},
        {"role": "user", "content": input_text},
    ])
    genre = response.choices[0].message.content.strip().lower()
    return genre if genre in ["fantasy", "sci-fi", "mystery"] else "mystery"

def generate_fantasy_story(state: State):
    """Creates a fantasy story."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Write a fantasy story based on the input."},
            {"role": "user", "content": state["input"]},
        ],
        max_tokens=500,
    )
    return {"output": response.choices[0].message.content.strip(), "decision": "Fantasy"}

def generate_sci_fi_story(state: State):
    """Creates a sci-fi story."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Write a sci-fi story based on the input."},
            {"role": "user", "content": state["input"]},
        ],
        max_tokens=500,
    )
    return {"output": response.choices[0].message.content.strip(), "decision": "Sci-Fi"}

def generate_mystery_story(state: State):
    """Creates a mystery story."""
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Write a mystery story based on the input."},
            {"role": "user", "content": state["input"]},
        ],
        max_tokens=500,
    )
    return {"output": response.choices[0].message.content.strip(), "decision": "Mystery"}

# Routing function to determine which story function to call
def route_request(state: State):
    """Determines the genre and routes accordingly."""
    decision = get_router_response(state["input"])
    return {"decision": decision}

def route_decision(state: State):
    """Maps the decision to the correct function."""
    return {
        "fantasy": "generate_fantasy_story",
        "sci-fi": "generate_sci_fi_story",
        "mystery": "generate_mystery_story",
    }.get(state["decision"], "generate_mystery_story")

# Build LangGraph workflow
def build_workflow():
    """Constructs the parallel workflow."""
    workflow = StateGraph(State)
    workflow.add_node("generate_fantasy_story", generate_fantasy_story)
    workflow.add_node("generate_sci_fi_story", generate_sci_fi_story)
    workflow.add_node("generate_mystery_story", generate_mystery_story)
    workflow.add_node("route_request", route_request)
    workflow.add_edge(START, "route_request")
    workflow.add_conditional_edges(
        "route_request",
        route_decision,
        {
            "generate_fantasy_story": "generate_fantasy_story",
            "generate_sci_fi_story": "generate_sci_fi_story",
            "generate_mystery_story": "generate_mystery_story",
        },
    )
    workflow.add_edge("generate_fantasy_story", END)
    workflow.add_edge("generate_sci_fi_story", END)
    workflow.add_edge("generate_mystery_story", END)
    return workflow.compile()

# Implement the Streamlit UI
def run_streamlit_app():
    """Creates an interactive UI for story generation."""
    st.title("Genre-Based Story Generator")
    user_input = st.text_input("Enter your story idea", "")
    if st.button("Generate Story"):
        if user_input:
            workflow = build_workflow()
            state = workflow.invoke({"input": user_input})
            st.subheader("Detected Genre:")
            st.write(state["decision"].capitalize())
            st.subheader("Generated Story:")
            st.write(state["output"])

if __name__ == "__main__":
    run_streamlit_app()