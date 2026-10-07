from pathlib import Path

from dotenv import load_dotenv
import os
import streamlit as st
from pydantic import BaseModel
from openai import OpenAI

# Load environment variables from the project folder, regardless of where the app is launched from
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not api_key or not endpoint or not deployment:
    st.error(
        "Missing Azure AI Foundry settings in .env. "
        "Check AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT."
    )
    st.stop()

# Foundry uses the OpenAI-compatible endpoint
client = OpenAI(
    api_key=api_key,
    base_url=endpoint.rstrip("/"),
    default_query={"api-version": api_version},
)

class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str

st.title("Complaint Entry")
st.write("Please fill out a complaint if you have one.")

# Section 1: Simple prompt
complaint_prompt = st.text_area("Enter your complaint", height=150)

if st.button("Generate"):
    if not complaint_prompt.strip():
        st.warning("Please enter a complaint first.")
    else:
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": complaint_prompt}],
            temperature=0.7,
        )
        st.subheader("Response")
        st.write(response.choices[0].message.content)

st.divider()

