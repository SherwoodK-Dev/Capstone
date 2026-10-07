# Step 1: Set Up the Environment
import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import openai

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

# Define the state structure
class State(TypedDict):
    topic: str
    advertisement: str
    review: str
    tagline: str
    combined_output: str

# Generate an advertisement
def generate_advertisement(state: State):
    """Calls OpenAI API to generate an advertisement related to the given topic."""
    msg = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a creative AI that writes catchy advertisements."},
            {"role": "user", "content": f"Write a catchy advertisement for a product related to {state['topic']}."}
        ],
        max_tokens=200
    )
    advertisement = msg.choices[0].message.content.strip()
    return {"advertisement": advertisement}

def generate_review(state: State):
    """Calls OpenAI API to generate a detailed product review for the given topic."""
    msg = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that writes detailed product reviews."},
            {"role": "user", "content": f"Write a product review for a product related to {state['topic']}. Include pros and cons."}
        ],
        max_tokens=300
    )
    review = msg.choices[0].message.content.strip()
    return {"review": review}

# Generate a catchy tagline
def generate_tagline(state: State):
    """Calls OpenAI API to generate a catchy tagline for the given topic."""
    msg = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a creative AI that generates catchy taglines."},
            {"role": "user", "content": f"Create a short, catchy tagline for a product related to {state['topic']}."}
        ],
        max_tokens=50
    )
    tagline = msg.choices[0].message.content.strip()
    return {"tagline": tagline}

# Combine all outputs
def combine_outputs(state: State):
    """Combines the advertisement, review, and tagline into a single structured output."""
    combined = f"Creative Output for {state['topic']}:\n\n"
    combined += f"ADVERTISEMENT:\n{state['advertisement']}\n\n"
    combined += f"REVIEW:\n{state['review']}\n\n"
    combined += f"TAGLINE:\n{state['tagline']}"
    return {"combined_output": combined}

# Build the LangGraph workflow
def build_workflow():
    """Constructs and compiles the LangGraph parallel workflow."""
    parallel_builder = StateGraph(State)
    # Adding independent nodes (tasks that can run in parallel)
    parallel_builder.add_node("generate_advertisement", generate_advertisement)
    parallel_builder.add_node("generate_review", generate_review)
    parallel_builder.add_node("generate_tagline", generate_tagline)
    parallel_builder.add_node("combine_outputs", combine_outputs)
    # Setting up parallel execution
    parallel_builder.add_edge(START, "generate_advertisement")
    parallel_builder.add_edge(START, "generate_review")
    parallel_builder.add_edge(START, "generate_tagline")
    parallel_builder.add_edge("generate_advertisement", "combine_outputs")
    parallel_builder.add_edge("generate_review", "combine_outputs")
    parallel_builder.add_edge("generate_tagline", "combine_outputs")
    parallel_builder.add_edge("combine_outputs", END)
    # Compile the workflow
    parallel_workflow = parallel_builder.compile()
    return parallel_workflow

# Streamlit UI
def run_streamlit_app():
    """Handles Streamlit UI interactions and workflow execution."""
    st.title("Creative Advertisement Generator")
    topic = st.text_input("Enter the topic:")
    if st.button("Generate Advertisement"):
        parallel_workflow = build_workflow()
        state = parallel_workflow.invoke({"topic": topic})
        st.subheader("Combined Creative Output:")
        st.write(state["combined_output"])

if __name__ == "__main__":
    run_streamlit_app()
